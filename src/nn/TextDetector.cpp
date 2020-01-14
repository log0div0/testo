
#include "TextDetector.hpp"
#include <iostream>
#include <algorithm>

extern unsigned char TextDetector_onnx[];
extern unsigned int TextDetector_onnx_len;

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

TextDetector::TextDetector() {
	session = LoadModel(TextDetector_onnx, TextDetector_onnx_len);
}

TextDetector::~TextDetector() {

}

std::vector<TextLine> TextDetector::detect(const stb::Image& image)
{
	if (!image.data) {
		return {};
	}

	run_nn(image);
	return find_textlines();
}

void TextDetector::run_nn(const stb::Image& image) {
	if ((in_w != image.width) ||
		(in_h != image.height))
	{
		in_c = 3;
		in_h = image.height;
		in_w = image.width;
		in_pad_h = nearest_n_times_div_by_2(in_h, 4);
		in_pad_w = nearest_n_times_div_by_2(in_w, 4);

		out_c = 2;
		out_h = in_h;
		out_w = in_w;
		out_pad_h = nearest_n_times_div_by_2(out_h, 4);
		out_pad_w = nearest_n_times_div_by_2(out_w, 4);

		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

		std::array<int64_t, 4> in_shape = {1, in_c, in_pad_h, in_pad_w};
		std::array<int64_t, 4> out_shape = {1, out_c, out_pad_h, out_pad_w};

		in.resize(in_c * in_pad_h * in_pad_w);
		out.resize(out_c * out_pad_h * out_pad_w);

		std::fill(in.begin(), in.end(), 0);
		std::fill(out.begin(), out.end(), 0);

		in_tensor = std::make_unique<Ort::Value>(
			Ort::Value::CreateTensor<float>(memory_info, in.data(), in.size(), in_shape.data(), in_shape.size()));
		out_tensor = std::make_unique<Ort::Value>(
			Ort::Value::CreateTensor<float>(memory_info, out.data(), out.size(), out_shape.data(), out_shape.size()));

		labelingWu = LabelingWu(out_w, out_h);
	}

	for (int y = 0; y < image.height; ++y) {
		for (int x = 0; x < image.width; ++x) {
			for (int c = 0; c < 3; ++c) {
				int src_index = y * image.width * image.channels + x * image.channels + c;
				int dst_index = c * in_pad_h * in_pad_w + y * in_pad_w + x;
				in[dst_index] = float(image.data[src_index]) / 255.0f;
			}
		}
	}

	const char* in_names[] = {"input"};
	const char* out_names[] = {"output"};

	session->Run(Ort::RunOptions{nullptr}, in_names, &*in_tensor, 1, out_names, &*out_tensor, 1);
}

std::vector<TextLine> TextDetector::find_textlines() {
	std::vector<Rect> words = find_words();
	std::vector<bool> visited_words(words.size(), false);
	std::vector<TextLine> textlines;
	for (int x = 0; x < out_w; ++x) {
		for (int y = 0; y < out_h; ++y) {
			uint16_t l = labelingWu.L[y*out_w + x];
			if (!l) {
				continue;
			}
			if (visited_words[l-1]) {
				continue;
			}
			TextLine textline;
			Rect a = words[l-1];
			visited_words[l-1] = true;
			textline.rect = a;
			Word word;
			word.rect = a;
			textline.words.push_back(word);
			while (true) {
textline_next:
				for (int x = a.right; (x <= (a.right + a.width()*2)) && (x < out_w); ++x) {
					for (int y = a.top; y <= a.bottom; ++y) {
						uint16_t l = labelingWu.L[y*out_w + x];
						if (!l) {
							continue;
						}
						if (visited_words[l-1]) {
							continue;
						}
						Rect b = words[l-1];
						for (int x = a.right; x <= b.left; ++x) {
							for (int y = std::max(a.top, b.top); y <= std::min(a.bottom, b.bottom); ++y) {
								if (out[out_pad_h*out_pad_w + y*out_pad_w + x] >= 0.75) {
									visited_words[l-1] = true;
									textline.rect |= b;
									Word word;
									word.rect = b;
									textline.words.push_back(word);
									a = b;
									goto textline_next;
								}
							}
						}
						goto textline_finish;
					}
				}
				goto textline_finish;
			}
textline_finish:
			for (size_t i = 1; i < textline.words.size(); ++i) {
				auto& a = textline.words[i-1].rect;
				auto& b = textline.words[i].rect;
				int border = (a.right + b.left) / 2;
				a.right = border;
				b.left = border;
			}
			textlines.push_back(textline);
		}
	}
	std::sort(textlines.begin(), textlines.end(), [](const TextLine& a, const TextLine& b) {
		return a.rect.top < b.rect.top;
	});
	return textlines;
}

std::vector<Rect> TextDetector::find_words() {
	for (int y = 0; y < out_h; ++y) {
		for (int x = 0; x < out_w; ++x) {
			labelingWu.I[y*out_w + x] = out[y*out_pad_w + x] >= .75;
		}
	}
	std::vector<Rect> words = labelingWu.run();
	for (size_t i = 0; i < words.size(); ++i) {
		words[i] = adjust_rect(words[i], 0.25);
	}
	return words;
}

Rect TextDetector::adjust_rect(const Rect& rect, float threshold) {
	Rect new_rect;

	{
		int32_t x = rect.left;
		float prev_max = 1;
		while (x > 0) {
			--x;
			float max = 0;
			for (int32_t y = rect.top; y <= rect.bottom; ++y) {
				if (max < out[y*out_pad_w + x]) {
					max = out[y*out_pad_w + x];
				}
			}
			if ((max < threshold) || (max > prev_max)) {
				++x;
				break;
			}
			prev_max = max;
		}
		new_rect.left = x;
	}
	{
		int32_t x = rect.right;
		float prev_max = 1;
		while (x < (out_w - 1)) {
			++x;
			float max = 0;
			for (int32_t y = rect.top; y <= rect.bottom; ++y) {
				if (max < out[y*out_pad_w + x]) {
					max = out[y*out_pad_w + x];
				}
			}
			if ((max < threshold) || (max > prev_max)) {
				--x;
				break;
			}
			prev_max = max;
		}
		new_rect.right = x;
	}
	{
		int32_t y = rect.top;
		float prev_max = 1;
		while (y > 0) {
			--y;
			float max = 0;
			for (int32_t x = rect.left; x <= rect.right; ++x) {
				if (max < out[y*out_pad_w + x]) {
					max = out[y*out_pad_w + x];
				}
			}
			if ((max < threshold) || (max > prev_max)) {
				++y;
				break;
			}
			prev_max = max;
		}
		new_rect.top = y;
	}
	{
		int32_t y = rect.bottom;
		float prev_max = 1;
		while (y < (out_h - 1)) {
			++y;
			float max = 0;
			for (int32_t x = rect.left; x <= rect.right; ++x) {
				if (max < out[y*out_pad_w + x]) {
					max = out[y*out_pad_w + x];
				}
			}
			if ((max < threshold) || (max > prev_max)) {
				--y;
				break;
			}
			prev_max = max;
		}
		new_rect.bottom = y;
	}

	return new_rect;
}

}
