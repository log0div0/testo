
#include <Random.hpp>
#include <clipp.h>

struct Example {
	stb::Image<stb::RGB> img;
	std::string fg;
	std::string bg;
};

const int HEIGHT = 32;

stb::RGB hsv2rgb(float h, float s, float v) {
	int i = std::floor(h * 6);
	float f = h * 6 - i;
	float p = v * (1 - s);
	float q = v * (1 - f * s);
	float t = v * (1 - (1 - f) * s);

	float r, g, b;

	switch (i % 6) {
		case 0: r = v, g = t, b = p; break;
		case 1: r = q, g = v, b = p; break;
		case 2: r = p, g = v, b = t; break;
		case 3: r = p, g = q, b = v; break;
		case 4: r = t, g = p, b = v; break;
		case 5: r = v, g = p, b = q; break;
		default:
			throw std::runtime_error("Should not be there");
	}

	return {
		(uint8_t)std::round(r * 255),
		(uint8_t)std::round(g * 255),
		(uint8_t)std::round(b * 255)
	};
}

struct Color {
	stb::RGB random_shade() {
		float H = float(random_int(h.first, h.second) % 360) / 360;
		float S = random_float(s.first, s.second);
		float V = random_float(v.first, v.second);
		return hsv2rgb(H, S, V);
	}
	std::string name;
	std::pair<int, int> h;
	std::pair<float, float> s;
	std::pair<float, float> v;
};

std::vector<Color> colors = {
	{"white",	{0, 360},	{.0, .05},	{.9, 1.}},
	{"gray",	{0, 360},	{.0, .05},	{.2, .8}},
	{"black",	{0, 360},	{.0, .05},	{.0, .1}},
	{"red",		{350, 370},	{.5, 1.},	{.5, 1.}},
	{"orange",	{30, 36},	{.5, 1.},	{.5, 1.}},
	{"yellow",	{52, 64},	{.5, 1.},	{.5, 1.}},
	{"green",	{97, 125},	{.5, 1.},	{.5, 1.}},
	{"cyan",	{173, 182},	{.5, 1.},	{.5, 1.}},
	{"blue",	{220, 250},	{.5, 1.},	{.5, 1.}},
	{"purple",	{264, 281},	{.5, 1.},	{.5, 1.}}
};

Color random_color() {
	return colors.at(random_int(colors.size()));
}

std::pair<Color, Color> random_color_pair() {
	Color fg = random_color();
	while (true) {
		Color bg = random_color();
		if (fg.name != bg.name) {
			return {fg, bg};
		}
	}
}

Example generate_example(int ratio) {
	std::unique_ptr<Font> font = random_font();
	Textline textline = random_textline(font, font->size() * ratio);
	int off_left = random_int(textline.bitmap.h / 2);
	int off_top = random_int(textline.bitmap.h / 2);
	int off_right = random_int(textline.bitmap.h / 2);
	int off_bottom = random_int(textline.bitmap.h / 2);
	std::pair<Color, Color> colors = random_color_pair();
	stb::Image<stb::RGB> img(
		off_left + textline.bitmap.w + off_right,
		off_top + textline.bitmap.h + off_bottom,
		colors.second.random_shade()
	);
	img.blend(off_left, off_top, colors.first.random_shade(), textline.bitmap);
	Example example;
	example.img = std::move(img);
	example.fg = colors.first.name;
	example.bg = colors.second.name;
	return example;
}

nlohmann::json generate_batch(int batch) {
	fs::path batch_dir = fs::path(out_dir) / std::to_string(batch);
	if (!fs::exists(batch_dir)) {
		fs::create_directories(batch_dir);
	}
	nlohmann::json batch_meta = {
		{"examples", nlohmann::json::array()}
	};
	int min_ratio = 2;
	int max_ratio = 60;
	int ratio = random_int(min_ratio, max_ratio);
	int mean_ratio = (min_ratio + max_ratio) / 2;
	int i = 0;
	int total_ratio = 0;
	do {
		Example example = generate_example(ratio);
		example.img = example.img.resize(HEIGHT * ratio * 2, HEIGHT);
		example.img.write_png(batch_dir / (std::to_string(i) + ".png"));
		batch_meta["examples"].push_back({
			{"img_path", std::to_string(batch) + "/" + std::to_string(i) + ".png"},
			{"fg", example.fg},
			{"bg", example.bg},
		});
		total_ratio += ratio;
		++i;
	} while (total_ratio < (batch_size * mean_ratio));
	return batch_meta;
}

int main(int argc, char** argv) {
	try {
		using namespace clipp;

		auto cli = (
			required("--input") & value("input dir", in_dir),
			required("--output") & value("output dir", out_dir),
			required("--batch_size") & value("batch size", batch_size),
			required("--batch_count") & value("batch count", batch_count)
		);

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			return 1;
		}

		load_fonts();
		run_batch_generation(generate_batch);

		return 0;
	} catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
		return 1;
	}
}
