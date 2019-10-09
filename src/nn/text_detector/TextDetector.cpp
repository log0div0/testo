
#include "TextDetector.hpp"
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <iostream>
#include <utf8.hpp>

extern unsigned char TextDetector_tflite[];
extern unsigned int TextDetector_tflite_len;

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
	"1",
	"2",
	"3Зз",
	"4",
	"5",
	"6Бб",
	"7",
	"8",
	"9",
	"AАaа",
	"bЬьЪъ",
	"Gg",
	"OОoо0",
	"YУyу",
	"Zz",
	"BВв",
	"CСcс",
	"Dd",
	"EЕЁeеё",
	"Ff",
	"HНhн",
	"Ii",
	"Jj",
	"KКkк",
	"Ll",
	"MМmм",
	"Nn",
	"PРpр",
	"Qq",
	"Rr",
	"Ss",
	"TТtт",
	"Uu",
	"Vv",
	"Ww",
	"XХxх",
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
	"Гг",
	"Дд",
	"Жж",
	"ИЙий",
	"Лл",
	"Пп",
	"Фф",
	"Цц",
	"Чч",
	"ШЩшщ",
	"Ыы",
	"Ээ",
	"Юю",
	"Яя"
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
	std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromBuffer(
		(const char*)TextDetector_tflite, TextDetector_tflite_len);
	if (!model) {
		throw std::runtime_error("Failed to load nn model");
	}

	tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder builder(*model, resolver);
	builder(&interpreter);
	if (!interpreter) {
		throw std::runtime_error("Failed to build nn model");
	}
}

TextDetector::~TextDetector() {

}

bool TextDetector::detect(stb::Image& image,
	const std::string& text,
	const std::string& foreground,
	const std::string& background)
{
	if (!image.data) {
		return false;
	}

	if ((in_w != image.width) ||
		(in_h != image.height))
	{
		in_w = image.width;
		in_h = image.height;
		int in_index = interpreter->inputs().at(0);
		if (interpreter->ResizeInputTensor(in_index, {1, in_h, in_w, 3}) != kTfLiteOk) {
			throw std::runtime_error("Failed to resize input tensor");
		}
		if (interpreter->AllocateTensors() != kTfLiteOk) {
			throw std::runtime_error("Failed to allocate tensors");
		}
		in = interpreter->typed_input_tensor<float>(0);
		out = interpreter->typed_output_tensor<float>(0);
		int out_index = interpreter->outputs().at(0);
		out_h = interpreter->tensor(out_index)->dims->data[1];
		out_w = interpreter->tensor(out_index)->dims->data[2];
		out_c = interpreter->tensor(out_index)->dims->data[3];
	}

	for (int y = 0; y < image.height; ++y) {
		for (int x = 0; x < image.width; ++x) {
			for (int c = 0; c < 3; ++c) {
				int src_index = y * image.width * image.channels + x * image.channels + c;
				int dst_index = y * image.width * 3 + x * 3 + c;
				in[dst_index] = float(image.data[src_index]) / 255.0f;
			}
		}
	}

	interpreter->Invoke();

	bool result = false;

	std::vector<std::string> query = utf8::split_to_chars(text);
	int foreground_id = get_color_id(foreground);
	int background_id = get_color_id(background);

	for (int y = 0; y < out_h; ++y) {
		for (int x = 0; x < out_w; ++x) {
			std::vector<Rect> rects;
			if (find_substr(x, y, query, 0, foreground_id, background_id, 0, 0, rects)) {
				result = true;
				for (auto& rect: rects) {
					image.draw(rect.left, rect.top, rect.right, rect.bottom, 200, 20, 50);
				}
			}

		}
	}

	return result;
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
		bottom += 2;
		if (query.at(index) != " ") {
			break;
		} else {
			++index;
		}
	}

	int symbol_id = get_symbol_id(query.at(index));

	for (int y = top; (y < bottom) && (y < out_h); ++y) {
		for (int x = left; (x < right) && (x < out_w); ++x) {
			int i = y * out_w * out_c + x * out_c;

			float objectness = out[i];
			if (objectness < 0.01f) {
				continue;
			}

			float class_probability = out[i + 1 + 2 + 2 + symbol_id];
			if (class_probability < 0.01f) {
				continue;
			}

			if (foreground_id >= 0) {
				float foreground_probability = out[i + 1 + 2 + 2 + symbols.size() + foreground_id];
				if (foreground_probability > 0.01f) {
					foreground_hits += 1;
				}
			}

			if (background_id >= 0) {
				float background_probability = out[i + 1 + 2 + 2 + symbols.size() + colors.size() + background_id];
				if (background_probability > 0.01f) {
					background_hits += 1;
				}
			}

			Box b;
			b.x = (x + out[i + 1]) / out_w;
			b.y = (y + out[i + 2]) / out_h;
			b.w = out[i + 3] * char_w / in_w;
			b.h = out[i + 4] * char_h / in_h;

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

			return find_substr(x + 1, top, query, index + 1, foreground_id, background_id, foreground_hits, background_hits, rects);
		}
	}
	return false;
}
