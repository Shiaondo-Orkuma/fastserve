#pragma once

#include "../core/tensor.h"

namespace fastserve {
namespace layers {

Tensor2D im2col(
    const Tensor3D& input,
    int C_in, int H, int W,
    int K_h, int K_w,
    int stride = 1,
    int padding = 0);

Tensor3D conv2d(
    const Tensor3D& input,
    const Tensor4D& weights,
    int C_in, int H, int W,
    int C_out, int K_h, int K_w,
    int stride = 1, int padding = 0);

void relu_inplace(Tensor3D& x);

Tensor3D relu(const Tensor3D& x);

Tensor3D maxpool2d(
    const Tensor3D& x,
    int pool_h, int pool_w,
    int stride_h, int stride_w);

Tensor1D flatten(const Tensor3D& x);

Tensor1D fully_connected(
    const Tensor1D& input,
    const Tensor2D& weights,
    const Tensor1D& bias);

Tensor1D softmax(const Tensor1D& logits);

}
}
