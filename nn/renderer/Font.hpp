
#include <stb/TrueType.hpp>
#include <psf/Font.hpp>
#include "Rect.hpp"

using ScaledHMetrics = stb::ScaledHMetrics;
using Char = stb::Char;

struct Font {
	virtual ~Font() = default;
	virtual float size() = 0;
	virtual ScaledHMetrics codepointHMetrics(char32_t codepoint) = 0;
	virtual float codepointKernAdvance(char32_t ch1, char32_t ch2) = 0;
	virtual Char renderCodepoint(char32_t codepoint, float xpos) = 0;
};

struct TTFont: Font {
	stb::ScaledFont font;
	bool use_antialiasing;
	float letter_spacing;

	TTFont(stb::ScaledFont font_, bool use_antialiasing_, float letter_spacing_):
		font(font_), use_antialiasing(use_antialiasing_), letter_spacing(letter_spacing_) {}

	virtual float size() override {
		return font.ascent() - font.descent();
	}

	virtual ScaledHMetrics codepointHMetrics(char32_t codepoint) override {
		return font.codepointHMetrics(codepoint);
	}

	virtual float codepointKernAdvance(char32_t ch1, char32_t ch2) override {
		return font.codepointKernAdvance(ch1, ch2) + letter_spacing;
	}

	virtual Char renderCodepoint(char32_t codepoint, float xpos) override {
		return font.renderCodepoint(codepoint, xpos, use_antialiasing);
	}
};

struct PSFFont: Font {
	const psf::Font& font;

	PSFFont(const psf::Font& font_): font(font_) {}

	virtual float size() override {
		return font.height();
	}

	virtual ScaledHMetrics codepointHMetrics(char32_t codepoint) override {
		ScaledHMetrics metrics;
		metrics.advanceWidth = font.width();
		metrics.leftSideBearing = 0;
		return metrics;
	}

	virtual float codepointKernAdvance(char32_t ch1, char32_t ch2) override {
		return 0;
	}

	static Rect get_bitmap_bbox(const stb::Image<uint8_t>& bitmap) {
		Rect bbox = {};
		for (int x = 0; x < bitmap.w; ++x) {
			for (int y = 0; y < bitmap.h; ++y) {
				if (bitmap.at(x, y) != 0) {
					bbox.x = x;
					goto right;
				}
			}
		}
	right:
		for (int x = bitmap.w - 1; x >= 0; --x) {
			for (int y = 0; y < bitmap.h; ++y) {
				if (bitmap.at(x, y) != 0) {
					bbox.w = x - bbox.x + 1;
					goto top;
				}
			}
		}
	top:
		for (int y = 0; y < bitmap.h; ++y) {
			for (int x = 0; x < bitmap.w; ++x) {
				if (bitmap.at(x, y) != 0) {
					bbox.y = y;
					goto bottom;
				}
			}
		}
	bottom:
		for (int y = bitmap.h - 1; y >= 0; --y) {
			for (int x = 0; x < bitmap.w; ++x) {
				if (bitmap.at(x, y) != 0) {
					bbox.h = y - bbox.y + 1;
					goto exit;
				}
			}
		}
	exit:
		return bbox;
	}

	virtual Char renderCodepoint(char32_t codepoint, float xpos) override {
		stb::Image<uint8_t> bitmap = font.codepointBitmap(codepoint);
		Rect bbox = get_bitmap_bbox(bitmap);
		Char ch(codepoint);
		ch.x = xpos + bbox.x;
		ch.y = bbox.y;
		ch.bitmap = stb::Image<stb::RGB>(bbox.w, bbox.h);
		for (int y = 0; y < bbox.h; ++y) {
			for (int x = 0; x < bbox.w; ++x) {
				ch.bitmap.at(x, y).r = bitmap.at(bbox.x + x, bbox.y + y);
				ch.bitmap.at(x, y).g = bitmap.at(bbox.x + x, bbox.y + y);
				ch.bitmap.at(x, y).b = bitmap.at(bbox.x + x, bbox.y + y);
			}
		}
		return ch;
	}
};
