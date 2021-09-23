
#pragma once

#include "Tensor.hpp"
#include "Img.hpp"
#include "ImgDetector.hpp"
#include <stb/Image.hpp>

#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;

namespace nn {

struct ImgTensor: Tensor<Img> {
	ImgTensor from_left(size_t i) const { return nn::from_left(*this, i); }
	ImgTensor from_top(size_t i) const { return nn::from_top(*this, i); }
	ImgTensor from_right(size_t i) const { return nn::from_right(*this, i); }
	ImgTensor from_bottom(size_t i) const { return nn::from_bottom(*this, i); }

};

template <typename RefType>
ImgTensor find_img(const stb::Image<stb::RGB>* image, RefType ref) {
	ImgTensor result;

	result.objects = ImgDetector::instance().detect(image, ref);

	std::sort(result.objects.begin(), result.objects.end(), [](const Img& a, const Img& b) {
		return a.rect.top < b.rect.top;
	});

	return result;
}

inline void to_json(nlohmann::json& j, const ImgTensor& tensor) {
	j["type"] = "ImgTensor";
	j["objects"] = nlohmann::json::array();

	for (auto& obj: tensor.objects) {
		j["objects"].push_back(obj);
	}
}

inline void from_json(const nlohmann::json& j, ImgTensor& tensor) {
	for (auto& i: j.at("objects")) {
		tensor.objects.push_back(i.get<Img>());
	}
}

}
