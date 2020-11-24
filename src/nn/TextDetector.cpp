
#include "TextDetector.hpp"
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

TextDetector& TextDetector::instance() {
	static TextDetector instance;
	return instance;
}

std::vector<TextLine> TextDetector::detect(const stb::Image<stb::RGB>* image)
{
	if (!image->data) {
		return {};
	}

	run_nn(image);
	return run_postprocessing();
}

void TextDetector::run_nn(const stb::Image<stb::RGB>* image) {
	if ((in_w != image->w) ||
		(in_h != image->h))
	{
		in_h = image->h;
		in_w = image->w;
		int in_pad_h = nearest_n_times_div_by_2(in_h, 4);
		int in_pad_w = nearest_n_times_div_by_2(in_w, 4);

		out_h = in_h;
		out_w = in_w;
		int out_pad_h = nearest_n_times_div_by_2(out_h, 4);
		int out_pad_w = nearest_n_times_div_by_2(out_w, 4);

		in.resize(in_pad_w, in_pad_h, 3);
		out.resize(out_pad_w, out_pad_h, 2);
		in.fill(0);
		out.fill(0);

		labeling_wu[0] = LabelingWu(out_w, out_h);
		labeling_wu[1] = LabelingWu(out_w, out_h);
	}

	in.set(*image, true);

	model.run({&in}, {&out});
}

std::vector<TextLine> TextDetector::run_postprocessing() {
	std::vector<Rect> up_rects = find_rects(0);
	std::vector<Rect> down_rects = find_rects(1);
	std::vector<TextLine> textlines;
	for (const Rect& up_rect: up_rects) {
		int x = up_rect.center_x();
		int y_begin = up_rect.bottom;
		int y_end = up_rect.bottom + up_rect.height();
		if (y_end > out_h) {
			y_end = out_h;
		}
		for (int y = y_begin; y < y_end; ++y) {
			uint16_t l = labeling_wu[1].L[y*out_w + x];
			if (l) {
				const Rect& down_rect = down_rects.at(l-1);
				Rect textline_rect = up_rect | down_rect;
				bool found = false;
				for (auto& textline: textlines) {
					if (textline.rect.iou(textline_rect) > 0.25) {
						textline.rect |= textline_rect;
						found = true;
						break;
					}
				}
				if (!found) {
					TextLine textline;
					textline.rect = textline_rect;
					textlines.push_back(textline);
				}
				break;
			}
		}
	}

	return textlines;
}

std::vector<Rect> TextDetector::find_rects(int c) {
	for (int y = 0; y < out_h; ++y) {
		for (int x = 0; x < out_w; ++x) {
			labeling_wu[c].I[y*out_w + x] = out.at(x, y, c) >= .75;
		}
	}
	std::vector<Rect> rects = labeling_wu[c].run();
	return rects;
}

}
