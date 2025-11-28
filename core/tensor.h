#pragma once

#include <vector>

namespace fastserve {

using Tensor1D = std::vector<float>;

using Tensor2D = std::vector<std::vector<float>>;

using Tensor3D = std::vector<std::vector<std::vector<float>>>;

using Tensor4D = std::vector<std::vector<std::vector<std::vector<float>>>>;

} 
