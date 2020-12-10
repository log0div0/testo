
#include <iostream>
#include "TrueType.hpp"
#include <clipp.h>
#include <codecvt>
#include <locale>

std::string english = "abcdefghijklmnopqrstuvwxyz";
std::string ENGLISH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
std::string russian = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя";
std::string RUSSIAN = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ";
std::string _alphabet = "0123456789!?\"'#$%&@()[]{}<>+-*/\\.,:;^~=|_" + english + ENGLISH + russian + RUSSIAN;

std::u32string to_utf32(const std::string &s) {
	std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
	return conv.from_bytes(s);
}

int main(int argc, char** argv) {
	try {
		using namespace clipp;

		std::string input_path;
		std::string output_path;
		bool antialiasing = false;

		auto cli = (
			required("--input") & value("input path", input_path),
			required("--output") & value("output path", output_path),
			option("--antialiasing").set(antialiasing)
		);

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			return 1;
		}

		stb::Font font(input_path);
		stb::Image<stb::RGB> img(1000, 800);
		img.fill(stb::RGB::white());

		int padding = 20;

		float y = padding;
		for (int font_size = 7; font_size < 20; ++font_size) {
			float x = padding;
			stb::ScaledFont scaled_font = font.scaleForPixelHeight(font_size);
			std::u32string u32alphabet = to_utf32(std::to_string(font_size) + " " + _alphabet);
			for (size_t i = 0; i < u32alphabet.size(); ++i) {
				if (' ' == u32alphabet[i]) {
					x += font_size / 2;
				}
				stb::ScaledHMetrics hmetrics = scaled_font.codepointHMetrics(u32alphabet[i]);
				if (i) {
					x += scaled_font.codepointKernAdvance(u32alphabet[i-1], u32alphabet[i]);
				}
				stb::Char ch = scaled_font.renderCodepoint(u32alphabet[i], x, antialiasing);
				if ((ch.x + ch.bitmap.w + padding) > img.w) {
					y += font_size + scaled_font.lineGap();
					x = padding;
					ch = scaled_font.renderCodepoint(u32alphabet[i], x, antialiasing);
				}
				img.blend(ch.x, y + ch.y, stb::RGB::black(), ch.bitmap);
				x += hmetrics.advanceWidth;
			}
			y += font_size + scaled_font.lineGap();
		}

		img.write_png(output_path);

		return 0;
	} catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
	}
}
