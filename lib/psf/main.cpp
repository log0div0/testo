
#include <iostream>
#include <psf/Font.hpp>
#include <clipp.h>
#include <codecvt>
#include <locale>

std::string english = "abcdefghijklmnopqrstuvwxyz";
std::string ENGLISH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
std::string russian = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя";
std::string RUSSIAN = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ";
std::string alphabet = "0123456789!?\"'#$%&@()[]{}<>+-*/\\.,:;^~=|_" + english + ENGLISH + russian + RUSSIAN;

std::u32string to_utf32(const std::string &s) {
	std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
	return conv.from_bytes(s);
}

int main(int argc, char** argv) {
	try {
		using namespace clipp;

		std::string input_path;
		std::string output_path;

		auto cli = (
			required("--input") & value("input path", input_path),
			required("--output") & value("output path", output_path)
		);

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			return 1;
		}

		int columns = 80;
		int rows = 40;

		psf::Font font(input_path);
		stb::Image<stb::RGB> img(font.width() * columns, font.height() * rows);
		img.fill(stb::RGB::white());

		int i = 0;
		for (char32_t ch: to_utf32(alphabet)) {
			stb::Image<uint8_t> bitmap = font.codepointBitmap(ch);
			int x = i % columns;
			int y = i / columns;
			img.blend(x * font.width(), y * font.height(), stb::RGB::black(), bitmap);
			++i;
		}

		img.write_png(output_path);

		return 0;
	} catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
	}
}
