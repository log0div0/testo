
#include "ImgDetector.hpp"
#include <iostream>
#include <algorithm>

namespace nn {

static inline bool is_n_times_div_by_2(int value, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		if ((value % 2) != 0) {
			return false;
		}
		value /= 2;
	}
	return true;
}

static inline int nearest_n_times_div_by_2(int value, size_t n) {
	while (true) {
		if (is_n_times_div_by_2(value, n)) {
			return value;
		}
		value += 1;
	}
}

ImgDetector& ImgDetector::instance() {
	static ImgDetector instance;
	return instance;
}

std::vector<Img> ImgDetector::detect(const stb::Image<stb::RGB>* image, const std::string& ref_img_path)
{
	if (!image->data) {
		return {};
	}

	run_nn(image, ref_img_path);
	return run_postprocessing();
}

#define REF_W 63
#define REF_H 63

void ImgDetector::run_nn(const stb::Image<stb::RGB>* srch_img, const std::string& new_ref_img_path) {
	if (
		(ref_img_path != new_ref_img_path) ||
		(srch_w != srch_img->w) ||
		(srch_h != srch_img->h)
	) {
		stb::Image<stb::RGB> ref_img(new_ref_img_path);

		ref_img_path = new_ref_img_path;
		srch_w = srch_img->w;
		srch_h = srch_img->h;

		ref_w = ref_img.w;
		ref_h = ref_img.h;

		ratio_w = float(REF_W) / ref_w;
		ratio_h = float(REF_H) / ref_h;

		SRCH_W = nearest_n_times_div_by_2(ratio_w * srch_w, 2);
		SRCH_H = nearest_n_times_div_by_2(ratio_h * srch_h, 2);

		out_w = SRCH_W / 4 - (REF_W / 4 - 1);
		out_h = SRCH_H / 4 - (REF_H / 4 - 1);

		srch.resize(SRCH_W, SRCH_H, 3);
		ref.resize(REF_W, REF_H, 3);
		out.resize(out_w, out_h, 1);

		labeling_wu = LabelingWu(out_w, out_h);

		ref.set(ref_img.resize(REF_W, REF_H), true);
	}

	srch.set(srch_img->resize(SRCH_W, SRCH_H), true);

	model.run({&srch, &ref}, {&out});
}

std::vector<Img> ImgDetector::run_postprocessing() {
	std::vector<Rect> rects = find_rects();
	std::vector<Img> result;
	for (auto& rect: rects) {
		std::cout << rect.area() << std::endl;
		if (rect.area() < 50) {
			continue;
		}
		int center_x = ((rect.center_x() + (REF_W / 4 - 1) / 2) * 4) / ratio_w;
		int center_y = ((rect.center_y() + (REF_H / 4 - 1) / 2) * 4) / ratio_h;
		Img img;
		img.rect.left = center_x - ref_w / 2;
		img.rect.top = center_y - ref_h / 2;
		img.rect.right = center_x + ref_w / 2;
		img.rect.bottom = center_y + ref_w / 2;
		img.rect = img.rect & Rect{0, 0, srch_w - 1, srch_h - 1};
		result.push_back(img);
	}
	return result;
}

std::vector<Rect> ImgDetector::find_rects() {
	for (int y = 0; y < out_h; ++y) {
		for (int x = 0; x < out_w; ++x) {
			labeling_wu.I[y*out_w + x] = out.at(x, y, 0) >= .9;
		}
	}
	std::vector<Rect> rects = labeling_wu.run();
	return rects;
}

}
