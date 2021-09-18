
#include "ImgDetector.hpp"
#include <iostream>
#include <algorithm>

namespace nn {

ImgDetector& ImgDetector::instance() {
	static ImgDetector instance;
	return instance;
}

bool is_sub_image_match(const stb::Image<stb::RGB>& img, const stb::Image<stb::RGB>& sub, int off_x, int off_y) {
	int different_pixels_count = 0;
	int max_different_pixels_count = (sub.h * sub.w) * 0.05f;
	for (int y = 0; y < sub.h; ++y) {
		for (int x = 0; x < sub.w; ++x) {
			if (img.at(off_x + x, off_y + y).max_channel_diff(sub.at(x, y)) > 8) {
				different_pixels_count += 1;
			}
			if (different_pixels_count > max_different_pixels_count) {
				return false;
			}
		}
	}
	return true;
}

std::vector<Img> ImgDetector::detect(const stb::Image<stb::RGB>* srch_img, const std::string& ref_img_path)
{
	stb::Image<stb::RGB> icon(ref_img_path);
	return detect(srch_img, &icon);
}

std::vector<Img> ImgDetector::detect(const stb::Image<stb::RGB>* srch_img, const stb::Image<stb::RGB>* ref_img)
{
	if (!srch_img->data) {
		return {};
	}

	std::vector<Img> result;
	int end_y = srch_img->h - ref_img->h + 1;
	int end_x = srch_img->w - ref_img->w + 1;
	for (int y = 0; y < end_y; ++y) {
		for (int x = 0; x < end_x; ++x) {
			if (is_sub_image_match(*srch_img, *ref_img, x, y)) {
				Img img;
				img.rect.left = x;
				img.rect.top = y;
				img.rect.right = x + ref_img->w - 1;
				img.rect.bottom = y + ref_img->h - 1;
				bool found = false;
				for (auto& other: result) {
					if (other.rect.iou(img.rect) >= 0.75f) {
						other.rect |= img.rect;
						found = true;
						break;
					}
				}
				if (!found) {
					result.push_back(img);
				}
			}
		}
	}
	return result;
}


}
