#pragma once

#include "../core/tensor.h"
#include <string>
#include <utility>

namespace fastserve {

class TinyCNN {
public:
    TinyCNN() = default;
    
    bool load(const std::string& path);
    
    bool is_loaded() const { return loaded_; }
    
    int predict(const Tensor3D& image) const;
    
    Tensor1D predict_probs(const Tensor3D& image) const;
    
    struct PredictionResult {
        int predicted_class;
        float confidence;
        Tensor1D probabilities;
    };
    PredictionResult predict_full(const Tensor3D& image) const;
    
    int num_parameters() const;
    std::string architecture_string() const;
    
private:
    bool loaded_ = false;
    
    static constexpr int C_IN = 1;
    static constexpr int H_IN = 28;
    static constexpr int W_IN = 28;
    static constexpr int C_OUT = 8;
    static constexpr int K_H = 3;
    static constexpr int K_W = 3;
    static constexpr int NUM_CLASSES = 10;
    
    Tensor4D conv_weights_;
    Tensor2D fc_weights_;
    Tensor1D fc_bias_;
};

}
