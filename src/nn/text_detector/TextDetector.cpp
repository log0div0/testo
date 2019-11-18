
#include "TextDetector.hpp"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <utf8.hpp>

extern unsigned char TextDetector_onnx[];
extern unsigned int TextDetector_onnx_len;

std::vector<std::string> colors = {
	"white",
	"gray",
	"black",
	"red",
	"orange",
	"yellow",
	"green",
	"cyan",
	"blue",
	"purple",
};

std::vector<std::string> symbols = {
	"0",
	"1",
	"2",
	"3",
	"4",
	"5",
	"6",
	"7",
	"8",
	"9",
	"Aa",
	"Bb",
	"Cc",
	"Dd",
	"Ee",
	"Ff",
	"Gg",
	"Hh",
	"Ii",
	"Jj",
	"Kk",
	"Ll",
	"Mm",
	"Nn",
	"Oo",
	"Pp",
	"Qq",
	"Rr",
	"Ss",
	"Tt",
	"Uu",
	"Vv",
	"Ww",
	"Xx",
	"Yy",
	"Zz",
	"!",
	"\"",
	"'",
	"`",
	"#",
	"$",
	"%",
	"&",
	"([{<",
	")]}>",
	"*",
	"+",
	".,",
	"-",
	"/",
	":;",
	"=",
	"?",
	"@",
	"\\",
	"^",
	"_",
	"|",
	"~",
	"Аа",
	"Бб",
	"Вв",
	"Гг",
	"Дд",
	"Ее",
	"Ёё",
	"Жж",
	"Зз",
	"Ии",
	"Йй",
	"Кк",
	"Лл",
	"Мм",
	"Нн",
	"Оо",
	"Пп",
	"Рр",
	"Сс",
	"Тт",
	"Уу",
	"Фф",
	"Хх",
	"Цц",
	"Чч",
	"Шш",
	"Щщ",
	"Ъъ",
	"Ыы",
	"Ьь",
	"Ээ",
	"Юю",
	"Яя",
};

int get_color_id(const std::string& color) {
	if (!color.size()) {
		return -1;
	}
	for (size_t i = 0; i < colors.size(); ++i) {
		if (colors[i] == color) {
			return i;
		}
	}
	throw std::runtime_error("Unknown color \"" + color + "\"");
}

int get_symbol_id(const std::string& symbol) {
	for (size_t i = 0; i < symbols.size(); ++i) {
		if (symbols[i].find(symbol) != std::string::npos) {
			return i;
		}
	}
	throw std::runtime_error("Unknown symbol \"" + symbol + "\"");
}

TextDetector::TextDetector() {
	env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "text_detector");
	session = std::make_unique<Ort::Session>(*env, TextDetector_onnx, TextDetector_onnx_len, Ort::SessionOptions{nullptr});
}

TextDetector::~TextDetector() {

}

std::vector<Rect> TextDetector::detect(stb::Image& image,
	const std::string& text,
	const std::string& foreground,
	const std::string& background)
{
	if (!image.data) {
		return {};
	}

	if ((in_w != image.width) ||
		(in_h != image.height))
	{
		in_c = 3;
		in_h = image.height;
		in_w = image.width;

		out_c = 1 + 2 + 2 + symbols.size() + colors.size() + colors.size();
		out_h = image.height / 8;
		out_w = image.width / 4;

		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

		std::array<int64_t, 4> in_shape = {1, in_c, in_h, in_w};
		std::array<int64_t, 4> out_shape = {1, out_c, out_h, out_w};

		in.resize(in_c * in_h * in_w);
		out.resize(out_c * out_h * out_w);

		in_tensor = std::make_unique<Ort::Value>(
			Ort::Value::CreateTensor<float>(memory_info, in.data(), in.size(), in_shape.data(), in_shape.size()));
    out_tensor = std::make_unique<Ort::Value>(
    	Ort::Value::CreateTensor<float>(memory_info, out.data(), out.size(), out_shape.data(), out_shape.size()));
	}

	for (int y = 0; y < image.height; ++y) {
		for (int x = 0; x < image.width; ++x) {
			for (int c = 0; c < 3; ++c) {
				int src_index = y * image.width * image.channels + x * image.channels + c;
				int dst_index = c * image.height * image.width + y * image.width + x;
				in[dst_index] = float(image.data[src_index]) / 255.0f;
			}
		}
	}

	const char* in_names[] = {"input"};
	const char* out_names[] = {"output"};

	session->Run(Ort::RunOptions{nullptr}, in_names, &*in_tensor, 1, out_names, &*out_tensor, 1);

	std::vector<std::string> query = utf8::split_to_chars(text);
	int foreground_id = get_color_id(foreground);
	int background_id = get_color_id(background);

	std::vector<Rect> total_rects;

	for (int y = 0; y < out_h; ++y) {
		for (int x = 0; x < out_w; ++x) {
			std::vector<Rect> rects;
			if (find_substr(x, y, query, 0, foreground_id, background_id, 0, 0, rects)) {
				Rect total_rect = rects.at(0);
				for (size_t i = 1; i < rects.size(); ++i) {
					total_rect |= rects.at(i);
				}
				total_rects.push_back(total_rect);
			}

		}
	}

	std::vector<Rect> results;
	for (auto& total_rect: total_rects) {
		bool merged = false;
		for (auto& result: results) {
			if (result.iou(total_rect) > 0.25f) {
				result |= total_rect;
				merged = true;
			}
		}
		if (!merged) {
			results.push_back(total_rect);
		}
	}
	return results;
}

float char_w = 8;
float char_h = 16;

bool TextDetector::find_substr(int left, int top,
	const std::vector<std::string>& query, size_t index,
	int foreground_id, int background_id,
	int foreground_hits, int background_hits,
	std::vector<Rect>& rects
) {
	int right = left;
	int bottom = top;
	bottom += 2;
	while (true) {
		if (index == query.size()) {
			if (foreground_id >= 0) {
				if (foreground_hits <= int(query.size() * 0.5f)) {
					return false;
				}
			}
			if (background_id >= 0) {
				if (background_hits <= int(query.size() * 0.5f)) {
					return false;
				}
			}
			return true;
		}
		right += 3;
		if (query.at(index) != " ") {
			break;
		} else {
			++index;
		}
	}

	int symbol_id = get_symbol_id(query.at(index));

	for (int y = top; (y < bottom) && (y < out_h); ++y) {
		for (int x = left; (x < right) && (x < out_w); ++x) {

			float objectness = at(x, y, 0);
			if (objectness < 0.10f) {
				continue;
			}

			float class_probability = at(x, y, 1 + 2 + 2 + symbol_id);
			if (class_probability < 0.10f) {
				continue;
			}

			if (foreground_id >= 0) {
				float foreground_probability = at(x, y, 1 + 2 + 2 + symbols.size() + foreground_id);
				if (foreground_probability > 0.10f) {
					foreground_hits += 1;
				}
			}

			if (background_id >= 0) {
				float background_probability = at(x, y, 1 + 2 + 2 + symbols.size() + colors.size() + background_id);
				if (background_probability > 0.10f) {
					background_hits += 1;
				}
			}

			Box b;
			b.x = (x + at(x, y, 1)) / out_w;
			b.y = (y + at(x, y, 2)) / out_h;
			b.w = at(x, y, 3) * char_w / in_w;
			b.h = at(x, y, 4) * char_h / in_h;

			Rect rect;
			rect.left = (b.x-b.w/2)*in_w;
			rect.right = (b.x+b.w/2)*in_w;
			rect.top = (b.y-b.h/2)*in_h;
			rect.bottom = (b.y+b.h/2)*in_h;

			if (rects.size()) {
				if (rects.back().iou(rect) >= 0.5f) {
					continue;
				}
			}

			rects.push_back(rect);

			if (find_substr(x + 1, top, query, index + 1, foreground_id, background_id, foreground_hits, background_hits, rects)) {
				return true;
			}
		}
	}
	return false;
}

float TextDetector::at(int x, int y, int c) {
	return out[c * out_h * out_w + y * out_w + x];
}