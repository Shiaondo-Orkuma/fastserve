#include "model.h"
#include "layers.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace fastserve {

bool TinyCNN::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open weights file: " << path << "\n";
        return false;
    }
    
    std::string line, token;
    
    std::getline(file, line);
    std::istringstream conv_header(line);
    int c_out, c_in, k_h, k_w;
    conv_header >> token >> c_out >> c_in >> k_h >> k_w;
    
    if (token != "CONV") {
        std::cerr << "Error: Expected 'CONV' header, got: " << token << "\n";
        return false;
    }
    
    conv_weights_.resize(c_out);
    for (int i = 0; i < c_out; i++) {
        conv_weights_[i].resize(c_in);
        for (int j = 0; j < c_in; j++) {
            conv_weights_[i][j].resize(k_h);
            for (int u = 0; u < k_h; u++) {
                conv_weights_[i][j][u].resize(k_w);
            }
        }
    }
    
    for (int co = 0; co < c_out; co++) {
        for (int ci = 0; ci < c_in; ci++) {
            for (int u = 0; u < k_h; u++) {
                for (int v = 0; v < k_w; v++) {
                    file >> conv_weights_[co][ci][u][v];
                }
            }
        }
    }
    
    file >> token;
    while (token != "FC" && file.good()) {
        file >> token;
    }
    
    int in_features, out_features;
    file >> in_features >> out_features;
    
    fc_weights_.resize(out_features);
    for (int i = 0; i < out_features; i++) {
        fc_weights_[i].resize(in_features);
    }
    
    for (int out_idx = 0; out_idx < out_features; out_idx++) {
        for (int in_idx = 0; in_idx < in_features; in_idx++) {
            file >> fc_weights_[out_idx][in_idx];
        }
    }
    
    file >> token;
    while (token != "BIAS" && file.good()) {
        file >> token;
    }
    
    int bias_count;
    file >> bias_count;
    
    fc_bias_.resize(bias_count);
    for (int i = 0; i < bias_count; i++) {
        file >> fc_bias_[i];
    }
    
    file.close();
    loaded_ = true;
    return true;
}

int TinyCNN::predict(const Tensor3D& image) const {
    auto result = predict_full(image);
    return result.predicted_class;
}

Tensor1D TinyCNN::predict_probs(const Tensor3D& image) const {
    auto result = predict_full(image);
    return result.probabilities;
}

TinyCNN::PredictionResult TinyCNN::predict_full(const Tensor3D& image) const {
    using namespace layers;
    
    auto conv_out = conv2d(image, conv_weights_, C_IN, H_IN, W_IN, C_OUT, K_H, K_W, 1, 0);
    
    relu_inplace(conv_out);
    
    auto pool_out = maxpool2d(conv_out, 2, 2, 2, 2);
    
    auto flat = flatten(pool_out);
    
    auto logits = fully_connected(flat, fc_weights_, fc_bias_);
    
    auto probs = softmax(logits);
    
    int predicted = 0;
    float max_prob = probs[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            predicted = i;
        }
    }
    
    return {predicted, max_prob, probs};
}

int TinyCNN::num_parameters() const {
    return 72 + 13520 + 10;
}

std::string TinyCNN::architecture_string() const {
    return "Conv(1->8, 3x3) -> ReLU -> MaxPool(2x2) -> FC(1352->10)";
}

}
