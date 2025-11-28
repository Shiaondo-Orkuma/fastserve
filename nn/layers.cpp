#include "layers.h"
#include "../core/simd_matmul.h"
#include <cmath>
#include <algorithm>

namespace fastserve {
namespace layers {

Tensor2D im2col(
    const Tensor3D& input,
    int C_in, int H, int W,
    int K_h, int K_w,
    int stride,
    int padding)
{
    int H_out = (H + 2 * padding - K_h) / stride + 1;
    int W_out = (W + 2 * padding - K_w) / stride + 1;
    
    int K = C_in * K_h * K_w;
    int N = H_out * W_out;
    
    Tensor2D X_col(K, std::vector<float>(N, 0.0f));
    
    for (int i = 0; i < H_out; i++) {
        for (int j = 0; j < W_out; j++) {
            int col = i * W_out + j;
            
            for (int c = 0; c < C_in; c++) {
                for (int u = 0; u < K_h; u++) {
                    for (int v = 0; v < K_w; v++) {
                        int row = c * (K_h * K_w) + u * K_w + v;
                        
                        int h_in = i * stride + u - padding;
                        int w_in = j * stride + v - padding;
                        
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            X_col[row][col] = input[c][h_in][w_in];
                        }
                    }
                }
            }
        }
    }
    
    return X_col;
}



Tensor3D conv2d(
    const Tensor3D& input,
    const Tensor4D& weights,
    int C_in, int H, int W,
    int C_out, int K_h, int K_w,
    int stride, int padding)
{
    Tensor2D X_col = im2col(input, C_in, H, W, K_h, K_w, stride, padding);
    
    int H_out = (H + 2 * padding - K_h) / stride + 1;
    int W_out = (W + 2 * padding - K_w) / stride + 1;
    int K = C_in * K_h * K_w;
    int N = H_out * W_out;
    
    Tensor2D W_mat(C_out, std::vector<float>(K));
    for (int c_out = 0; c_out < C_out; c_out++) {
        int k_idx = 0;
        for (int c_in = 0; c_in < C_in; c_in++) {
            for (int u = 0; u < K_h; u++) {
                for (int v = 0; v < K_w; v++) {
                    W_mat[c_out][k_idx++] = weights[c_out][c_in][u][v];
                }
            }
        }
    }
    
    Tensor2D Y_flat(C_out, std::vector<float>(N));
    matmul_blocked_simd(W_mat, X_col, Y_flat);
    
    Tensor3D output(C_out, std::vector<std::vector<float>>(H_out, std::vector<float>(W_out)));
    for (int c = 0; c < C_out; c++) {
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                output[c][i][j] = Y_flat[c][i * W_out + j];
            }
        }
    }
    
    return output;
}



void relu_inplace(Tensor3D& x) {
    for (auto& channel : x) {
        for (auto& row : channel) {
            for (auto& val : row) {
                if (val < 0) val = 0;
            }
        }
    }
}

Tensor3D relu(const Tensor3D& x) {
    Tensor3D result = x;
    relu_inplace(result);
    return result;
}


Tensor3D maxpool2d(
    const Tensor3D& x,
    int pool_h, int pool_w,
    int stride_h, int stride_w)
{
    int C = x.size();
    int H = x[0].size();
    int W = x[0][0].size();
    
    int H_out = (H - pool_h) / stride_h + 1;
    int W_out = (W - pool_w) / stride_w + 1;
    
    Tensor3D output(C, std::vector<std::vector<float>>(H_out, std::vector<float>(W_out)));
    
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                int h_start = i * stride_h;
                int w_start = j * stride_w;
                
                float max_val = x[c][h_start][w_start];
                for (int u = 0; u < pool_h; u++) {
                    for (int v = 0; v < pool_w; v++) {
                        max_val = std::max(max_val, x[c][h_start + u][w_start + v]);
                    }
                }
                output[c][i][j] = max_val;
            }
        }
    }
    
    return output;
}


Tensor1D flatten(const Tensor3D& x) {
    int C = x.size();
    int H = x[0].size();
    int W = x[0][0].size();
    
    Tensor1D output(C * H * W);
    int idx = 0;
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                output[idx++] = x[c][i][j];
            }
        }
    }
    return output;
}

Tensor1D fully_connected(
    const Tensor1D& input,
    const Tensor2D& weights,
    const Tensor1D& bias)
{
    int out_features = weights.size();
    int in_features = weights[0].size();
    
    Tensor1D output(out_features);
    
    for (int i = 0; i < out_features; i++) {
        float sum = bias[i];
        for (int j = 0; j < in_features; j++) {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum;
    }
    
    return output;
}


Tensor1D softmax(const Tensor1D& logits) {
    int n = logits.size();
    Tensor1D probs(n);
    
    float max_val = *std::max_element(logits.begin(), logits.end());
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    
    for (int i = 0; i < n; i++) {
        probs[i] /= sum;
    }
    
    return probs;
}

}
} 
