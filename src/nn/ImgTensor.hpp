
#pragma once

#include "Tensor.hpp"
#include "Img.hpp"
#include <stb/Image.hpp>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

namespace nn {

struct ImgTensor: Tensor<Img> {
	ImgTensor from_left(size_t i) const { return nn::from_left(*this, i); }
	ImgTensor from_top(size_t i) const { return nn::from_top(*this, i); }
	ImgTensor from_right(size_t i) const { return nn::from_right(*this, i); }
	ImgTensor from_bottom(size_t i) const { return nn::from_bottom(*this, i); }

};

ImgTensor find_img(const stb::Image<stb::RGB>* image, const fs::path& path_to_img);

}
