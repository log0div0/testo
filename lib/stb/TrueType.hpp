
#pragma once

#include <stb_truetype.h>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include "Image.hpp"

namespace stb {

struct Char {
	Char(char32_t codepoint_): codepoint(codepoint_) {}

	char32_t codepoint;
	int x = 0;
	int y = 0;
	int end_x() const {
		return x + bitmap.w;
	}
	int end_y() const {
		return y + bitmap.h;
	}
	stb::Image<stb::RGB> bitmap;
};

struct ScaledHMetrics {
	float advanceWidth = 0;
	float leftSideBearing = 0;
};

struct BitmapBox {
	int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};

inline BitmapBox operator|(const BitmapBox& a, const BitmapBox& b) {
	BitmapBox box;
	box.x0 = std::min(a.x0, b.x0);
	box.y0 = std::min(a.y0, b.y0);
	box.x1 = std::max(a.x1, b.x1);
	box.y1 = std::max(a.y1, b.y1);
	return box;
}

struct ScaledFont {
	ScaledFont(stbtt_fontinfo info, float scale): _info(info), _scale(scale) {
		int ascent = 0, descent = 0, lineGap = 0;
		stbtt_GetFontVMetrics(&_info, &ascent, &descent, &lineGap);
		_ascent = ascent * _scale;
		_descent = descent * _scale;
		_lineGap = lineGap * _scale;
	}

	float ascent() const {
		return _ascent;
	}

	float descent() const {
		return _descent;
	}

	float lineGap() const {
		return _lineGap;
	}

	ScaledHMetrics codepointHMetrics(char32_t codepoint) {
		int advanceWidth = 0;
		int leftSideBearing = 0;
		stbtt_GetCodepointHMetrics(&_info, codepoint, &advanceWidth, &leftSideBearing);

		ScaledHMetrics metrics;
		metrics.advanceWidth = advanceWidth * _scale;
		metrics.leftSideBearing = leftSideBearing * _scale;
		return metrics;
	}

	BitmapBox getCodepointBitmapBox(char32_t codepoint) {
		int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
		stbtt_GetCodepointBitmapBox(&_info, codepoint, _scale, _scale, &x0, &y0, &x1, &y1);

		BitmapBox box;
		box.x0 = x0;
		box.y0 = y0;
		box.x1 = x1;
		box.y1 = y1;
		return box;
	}

	BitmapBox getCodepointBitmapBoxSubpixel(char32_t codepoint, float shift_x, float shift_y) {
		int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
		stbtt_GetCodepointBitmapBoxSubpixel(&_info, codepoint, _scale, _scale, shift_x, shift_y, &x0, &y0, &x1, &y1);

		BitmapBox box;
		box.x0 = x0;
		box.y0 = y0;
		box.x1 = x1;
		box.y1 = y1;
		return box;
	}

	float codepointKernAdvance(char32_t ch1, char32_t ch2) {
		int advance = stbtt_GetCodepointKernAdvance(&_info, ch1, ch2);
		return advance * _scale;
	}

	void makeCodepointBitmap(unsigned char *output, int out_w, int out_h, int out_stride, int codepoint) {
		stbtt_MakeCodepointBitmap(&_info, output, out_w, out_h, out_stride, _scale, _scale, codepoint);
	}

	void makeCodepointBitmapSubpixel(unsigned char *output, int out_w, int out_h, int out_stride, float shift_x, float shift_y, int codepoint) {
		stbtt_MakeCodepointBitmapSubpixel(&_info, output, out_w, out_h, out_stride, _scale, _scale, shift_x, shift_y, codepoint);
	}

	Char renderCodepoint(char32_t codepoint, float xpos, bool use_antialiasing) {
		float shifts[3];
		BitmapBox boxes[3];
		for (int c = 0; c < 3; ++c) {
			if (use_antialiasing) {
				shifts[c] = (xpos + (2 - c) * 0.33f) - std::floor(xpos);
			} else {
				shifts[c] = xpos - std::floor(xpos);
			}
			boxes[c] = getCodepointBitmapBoxSubpixel(codepoint, shifts[c], 0);
		}
		BitmapBox box = boxes[0] | boxes[1] | boxes[2];
		Char ch(codepoint);
		ch.x = box.x0 + std::floor(xpos);
		ch.y = box.y0 + std::floor(ascent());
		ch.bitmap = stb::Image<stb::RGB>(box.x1 - box.x0, box.y1 - box.y0);
		ch.bitmap.fill({0,0,0});
		for (int c = 0; c < 3; ++c) {
			int off_x = boxes[c].x0 - box.x0;
			int off_y = boxes[c].y0 - box.y0;
			stb::Image<uint8_t> bitmap(boxes[c].x1 - boxes[c].x0, boxes[c].y1 - boxes[c].y0);
			bitmap.fill(0);
			makeCodepointBitmapSubpixel(&bitmap.at(0, 0), bitmap.w, bitmap.h, bitmap.stride(), shifts[c], 0, codepoint);
			for (int y = 0; y < bitmap.h; ++y) {
				for (int x = 0; x < bitmap.w; ++x) {
					ch.bitmap.at(off_x + x, off_y + y)[c] = bitmap.at(x, y);
				}
			}
		}
		return ch;
	}

private:
	float _ascent, _descent, _lineGap;
	stbtt_fontinfo _info = {};
	float _scale = 0;
};

struct Font {
	Font(const std::string& path);
	~Font();

	Font(const Font& other) = delete;
	Font& operator=(const Font& other) = delete;

	Font(Font&& other): buf(std::move(other.buf)), info(other.info) {
		other.info = {};
	}
	Font& operator=(Font&& other) {
		std::swap(buf, other.buf);
		std::swap(info, other.info);
		return *this;
	}

	ScaledFont scaleForPixelHeight(float h) const {
		float scale = stbtt_ScaleForPixelHeight(&info, h);
		return ScaledFont(info, scale);
	}

private:
	std::vector<uint8_t> buf;
	stbtt_fontinfo info = {};
};

}
