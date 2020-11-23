
#pragma once

#include "Tensor.hpp"
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

namespace nn {

TextTensor find_text(const stb::Image<stb::RGB>* image);
ImgTensor find_img(const stb::Image<stb::RGB>* image, const fs::path& path_to_img);

}
