#include "caffe2/operators/collect_and_distribute_fpn_rpn_proposals_op.h"

#include "caffe2/core/context_gpu.h"

#include <cub/device/device_radix_sort.cuh>

namespace caffe2 {

namespace {

// Compute the area of a box.
__device__ float box_area(float x1, float y1, float x2, float y2) {
    float w = x2 - x1 + 1;
    float h = y2 - y1 + 1;
    float area = w * h;
    return area;
}

// Determine which FPN level an RoI should map to based
// on the heuristics in the FPN paper.
__device__ int map_roi_to_fpn_levels(
    float x1, float y1, float x2, float y2,
    float k_min, float k_max,
    float roi_canonical_scale, float roi_canonical_level) {

    // Compute level id
    float s = sqrt(box_area(x1, y1, x2, y2));

    // Eqn.(1) in FPN paper
    int target_lvl = (int) floorf(
        roi_canonical_level +
        log2f(s / roi_canonical_scale + 1e-6));
    target_lvl = (int) fminf(fmaxf(target_lvl, k_min), k_max);
    return target_lvl;
}

__global__ void calculate_fpn_levels_kernel(
    const float* rois,
    int* lvls,
    int* lvls_freq,
    const int size,
    const int stride,
    const int roi_min_level,
    const int roi_max_level,
    const int roi_canonical_scale,
    const int roi_canonical_level) {

  for (int i = threadIdx.x + (blockIdx.x * blockDim.x);
      i < size; i += (blockDim.x * gridDim.x)) {
    const float* roi = &rois[i * stride];
    int level = map_roi_to_fpn_levels(
        roi[1], roi[2], roi[3], roi[4],
        roi_min_level, roi_max_level,
        roi_canonical_scale, roi_canonical_level);
    lvls[i] = level;
    atomicAdd(&lvls_freq[level - roi_min_level], 1);
  }
}

__global__ void move_rois_kernel(
    const float* rois,
    const int* sorted_indexes,
    float* rois_sorted,
    const int input_size,
    const int output_size,
    const int stride) {
  for (int i = threadIdx.x + (blockIdx.x * blockDim.x);
      i < input_size; i += (blockDim.x * gridDim.x)) {
    if (i < output_size) {
      for (int j = 0; j < stride; j++) {
        rois_sorted[i * stride + j] = rois[sorted_indexes[i] * stride + j];
      }
    }
  }
}

template<typename T>
void cub_sort(
    const Tensor<CUDAContext>& keys_in, Tensor<CUDAContext>& keys_out,
    const Tensor<CUDAContext>& values_in, Tensor<CUDAContext>& values_out,
    const int num_items,
    const bool descending,
    const CUDAContext& context) {
  size_t temp_storage_bytes = 0;

  if (descending) {
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
      NULL, temp_storage_bytes,
      keys_in.data<T>(), keys_out.mutable_data<T>(),
      values_in.data<int>(), values_out.mutable_data<int>(),
      num_items,
      0, sizeof(T) * 8,
      context.cuda_stream()));
  }
  else {
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      NULL, temp_storage_bytes,
      keys_in.data<T>(), keys_out.mutable_data<T>(),
      values_in.data<int>(), values_out.mutable_data<int>(),
      num_items,
      0, sizeof(T) * 8,
      context.cuda_stream()));
  }

  CHECK_GT(temp_storage_bytes, 0);
  Tensor<CUDAContext> d_temp_storage;
  d_temp_storage.Resize(temp_storage_bytes);

  if (descending) {
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
      (void*) d_temp_storage.mutable_data<char>(), temp_storage_bytes,
      keys_in.data<T>(), keys_out.mutable_data<T>(),
      values_in.data<int>(), values_out.mutable_data<int>(),
      num_items,
      0, sizeof(T) * 8,
      context.cuda_stream()));
  }
  else {
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      (void*) d_temp_storage.mutable_data<char>(), temp_storage_bytes,
      keys_in.data<T>(), keys_out.mutable_data<T>(),
      values_in.data<int>(), values_out.mutable_data<int>(),
      num_items,
      0, sizeof(T) * 8,
      context.cuda_stream()));
  }
}

} // namespace

template <>
bool CollectAndDistributeFpnRpnProposalsOp<CUDAContext>::RunOnDevice() {

  int num_rpn_lvls = rpn_max_level_ - rpn_min_level_ + 1;
  CAFFE_ENFORCE_EQ(InputSize(), 2 * num_rpn_lvls);

  int num_roi_lvls = roi_max_level_ - roi_min_level_ + 1;
  CAFFE_ENFORCE_EQ(OutputSize(), num_roi_lvls + 2);

  // Collect rois and scores in Tensor<CUDAContext>
  // rois are in [[batch_idx, x0, y0, x1, y2], ...] format
  // Combine predictions across all levels and retain the top scoring
  //
  // equivalent to python code
  //   roi_inputs = inputs[:num_rpn_lvls]
  //   score_inputs = inputs[num_rpn_lvls:]
  //   rois = np.concatenate([blob.data for blob in roi_inputs])
  //   scores = np.concatenate([blob.data for blob in score_inputs]).squeeze()
  int proposal_num = 0;
  for (int i = 0; i < num_rpn_lvls; i++) {
    const auto& roi_in = Input(i);
    proposal_num += roi_in.dim(0);
  }
  // in case rpn_post_nms_topN_ is larger than proposal_num
  auto post_nms_topN = std::min(proposal_num, rpn_post_nms_topN_);
  // concatenate all rois and their scores
  Tensor<CUDAContext> rois;
  rois.Resize(proposal_num, 5);
  Tensor<CUDAContext> scores;
  scores.Resize(proposal_num);
  int len = 0;
  for (int i = 0; i < num_rpn_lvls; i++) {
    const auto& roi_in = Input(i);
    const int n = roi_in.dim(0);
    context_.Copy<float, CUDAContext, CUDAContext>(
        n * 5,
        roi_in.data<float>(),
        rois.mutable_data<float>() + (len * 5));
    const auto& score_in = Input(num_rpn_lvls + i);
    context_.Copy<float, CUDAContext, CUDAContext>(
        n,
        score_in.data<float>(),
        scores.mutable_data<float>() + len);
    len += n;
  }

  // sort rois according to scores
  // creat a device array with values [0...proposal_num-1]
  std::vector<int> h_range_sorted(proposal_num);
  std::iota(h_range_sorted.begin(), h_range_sorted.end(), 0);
  Tensor<CUDAContext> d_range_sorted;
  d_range_sorted.Resize(proposal_num);
  CUDA_CHECK(cudaMemcpyAsync(
      d_range_sorted.mutable_data<int>(),
      h_range_sorted.data(),
      proposal_num * sizeof(int),
      cudaMemcpyDefault,
      context_.cuda_stream()));
  Tensor<CUDAContext> rois_indexes;
  rois_indexes.ResizeLike(d_range_sorted);
  Tensor<CUDAContext> scores_sorted;
  scores_sorted.ResizeLike(scores);
  // sort key-value pairs of (scores, d_range_sorted) according to scores
  cub_sort<float>(
    scores, scores_sorted,
    d_range_sorted, rois_indexes,
    proposal_num,
    true,
    context_);

  // move rois to their right position according to their sorted scores
  auto* rois_out = Output(0);
  rois_out->Resize(post_nms_topN, 5);
  move_rois_kernel<<<
      std::min(proposal_num / CAFFE_CUDA_NUM_THREADS, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS, 0,
      context_.cuda_stream()>>>(
          rois.mutable_data<float>(),
          rois_indexes.mutable_data<int>(),
          rois_out->mutable_data<float>(),
          proposal_num, post_nms_topN,
          5);

  // calculate the level of each roi
  // initialize d_lvls_freq with zeros
  std::vector<int> h_lvls_freq(num_roi_lvls, 0);
  Tensor<CUDAContext> d_lvls_freq;
  d_lvls_freq.Resize(num_roi_lvls);
  CUDA_CHECK(cudaMemcpyAsync(
      d_lvls_freq.mutable_data<int>(),
      h_lvls_freq.data(),
      num_roi_lvls * sizeof(int),
      cudaMemcpyDefault,
      context_.cuda_stream()));
  Tensor<CUDAContext> lvls;
  lvls.Resize(post_nms_topN);
  calculate_fpn_levels_kernel<<<
      std::min(proposal_num / CAFFE_CUDA_NUM_THREADS, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS, 0,
      context_.cuda_stream()>>>(
          rois_out->data<float>(),
          lvls.mutable_data<int>(),
          d_lvls_freq.mutable_data<int>(),
          post_nms_topN, 5,
          roi_min_level_, roi_max_level_,
          roi_canonical_scale_, roi_canonical_level_);
  // copy d_lvls_freq into host to be used later in distribution
  CUDA_CHECK(cudaMemcpyAsync(
      h_lvls_freq.data(),
      d_lvls_freq.data<int>(),
      num_roi_lvls * sizeof(int),
      cudaMemcpyDefault,
      context_.cuda_stream()));

  // sort rois according to their levels to facilitate distribution
  Tensor<CUDAContext> lvls_sorted;
  lvls_sorted.ResizeLike(lvls);
  cub_sort<int>(
    lvls, lvls_sorted,
    d_range_sorted, rois_indexes,
    post_nms_topN,
    false,
    context_);

  // move rois to the position according to their levels
  Tensor<CUDAContext> rois_level_sorted;
  rois_level_sorted.ResizeLike(*rois_out);
  move_rois_kernel<<<
      std::min(proposal_num / CAFFE_CUDA_NUM_THREADS, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS, 0,
      context_.cuda_stream()>>>(
          rois_out->data<float>(),
          rois_indexes.data<int>(),
          rois_level_sorted.mutable_data<float>(),
          post_nms_topN, post_nms_topN,
          5);

  // prepare rois_idx_restore
  auto* rois_idx_restore = Output(OutputSize() - 1);
  rois_idx_restore->Resize(post_nms_topN);
  // sort key-value pairs
  CUDA_CHECK(cudaMemcpyAsync(
      rois_idx_restore->mutable_data<int>(),
      h_range_sorted.data(),
      post_nms_topN * sizeof(int),
      cudaMemcpyDefault,
      context_.cuda_stream()));
  // only need the output values from the sort function
  // but still need to provide an output array for sorted keys
  Tensor<CUDAContext> dev_tmp;
  dev_tmp.Resize(post_nms_topN);
  cub_sort<int>(
    rois_indexes, dev_tmp,
    d_range_sorted, *rois_idx_restore,
    post_nms_topN,
    false,
    context_);

  // distribute rois based on levels (memcpy)
  auto rois_level_sorted_ptr = rois_level_sorted.data<float>();
  for (int i = 0, lvl = roi_min_level_; i < num_roi_lvls; i++, lvl++) {
      auto* roi_out = Output(i + 1);
      auto roi_out_size = h_lvls_freq[i];
      roi_out->Resize(roi_out_size, 5);
      CUDA_CHECK(cudaMemcpyAsync(
        roi_out->mutable_data<float>(),
        rois_level_sorted_ptr,
        roi_out_size * 5 * sizeof(float),
        cudaMemcpyDefault,
        context_.cuda_stream()));
      rois_level_sorted_ptr += roi_out_size * 5;
  }

  return true;
}

REGISTER_CUDA_OPERATOR(
    CollectAndDistributeFpnRpnProposals,
    CollectAndDistributeFpnRpnProposalsOp<CUDAContext>);

} // namespace caffe2
