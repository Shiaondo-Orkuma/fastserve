#pragma once

#include "tensor.h"
#include <immintrin.h>

namespace fastserve {

constexpr int BLOCK_SIZE = 32;

inline void matmul_blocked_simd(const Tensor2D& A, const Tensor2D& B, Tensor2D& C) {
    int M = A.size();
    int K = A[0].size();
    int N = B[0].size();
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0f;
        }
    }
    
    for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                int i_end = std::min(ii + BLOCK_SIZE, M);
                int k_end = std::min(kk + BLOCK_SIZE, K);
                int j_end = std::min(jj + BLOCK_SIZE, N);
                
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        float a_ik = A[i][k];
                        __m256 a_vec = _mm256_set1_ps(a_ik);
                        
                        int j = jj;
                        for (; j + 7 < j_end; j += 8) {
                            __m256 b_vec = _mm256_loadu_ps(&B[k][j]);
                            __m256 c_vec = _mm256_loadu_ps(&C[i][j]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&C[i][j], c_vec);
                        }
                        for (; j < j_end; j++) {
                            C[i][j] += a_ik * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

}
