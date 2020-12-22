
#include <Random.hpp>
#include <clipp.h>

struct TextObj {
	const stb::Image<stb::RGB>& screenshot;
	Rect bbox;
	std::string text;

	stb::Image<stb::RGB> random_crop() const {
		Rect crop = bbox;

		crop.extend_left(random_int(bbox.h / 2));
		crop.extend_top(random_int(bbox.h / 2));
		crop.extend_right(random_int(bbox.h / 2));
		crop.extend_bottom(random_int(bbox.h / 2));

		crop = crop & Rect(screenshot);

		return screenshot.sub_image(crop.x, crop.y, crop.w, crop.h);
	}
};

std::map<int, std::vector<TextObj>> text_objs;
std::vector<int> all_ratios;

struct Example {
	stb::Image<stb::RGB> img;
	std::string text;
};

bool is_unknown_codepoint(char32_t codepoint) {
	if (codepoint == space_codepoint()) {
		return false;
	}
	auto it = std::find(u32alphabet.begin(), u32alphabet.end(), codepoint);
	return it == u32alphabet.end();
}

bool contains_unknown_codepoint(const std::string& str) {
	for (char32_t codepoint: to_utf32(str)) {
		if (is_unknown_codepoint(codepoint)) {
			return true;
		}
	}
	return false;
}

void load_text_objs() {
	for (auto& doc: docs) {
		for (auto& obj: doc.meta.at("objs")) {
			if (obj.at("type") != "text") {
				continue;
			}
			Rect bbox = obj;
			std::string text = obj.at("text");
			bool ignore =
				(text.size() < 2) ||
				(contains_unknown_codepoint(text)) ||
				(bbox.w <= bbox.h) ||
				(bbox.h <= 10);
			if (ignore) {
				continue;
			}
			int ratio = ceilf(float(bbox.w) / float(bbox.h));
			text_objs[ratio].push_back({doc.img, bbox, text});
		}
	}
	for (auto& kv: text_objs) {
		all_ratios.push_back(kv.first);
	}
}

int random_ratio() {
	return all_ratios.at(random_int(all_ratios.size()));
}

const int HEIGHT = 32;

Example generate_real_example(int ratio, const TextObj& textline) {
	Example example;
	example.img = textline.random_crop();
	random_channels_shuffle(example.img);
	random_inverse(example.img);
	example.text = textline.text;
	return example;
}

stb::Image<stb::RGB> random_solid_bg(int w, int h) {
	return {w, h, random_RGB()};
}

stb::Image<stb::RGB> random_noise_bg(int w, int h) {
	float noise_prob = random_float(0, 1);
	stb::RGB color = random_RGB();
	stb::Image<stb::RGB> img(w, h);
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			if (random_bool(noise_prob)) {
				img.at(x, y) = random_RGB(color, 0, 32);
			} else {
				img.at(x, y) = color;
			}
		}
	}
	return img;
}

stb::Image<stb::RGB> random_gradient_bg(int w, int h) {
	stb::Image<stb::RGB> img(w, h);
	stb::RGB color_begin = random_RGB();
	stb::RGB color_end = random_RGB(color_begin, 16, 48);
	float r = color_begin.r;
	float g = color_begin.g;
	float b = color_begin.b;
	float r_step = float(color_end.r - color_begin.r) / h;
	float g_step = float(color_end.g - color_begin.g) / h;
	float b_step = float(color_end.b - color_begin.b) / h;
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			img.at(x, y).r = r + (r_step * y);
			img.at(x, y).g = g + (g_step * y);
			img.at(x, y).b = b + (b_step * y);
		}
	}
	return img;
}

stb::Image<stb::RGB> random_bg(int w, int h) {
	switch (random_int(3)) {
		case 0:
			return random_solid_bg(w, h);
		case 1:
			return random_noise_bg(w, h);
		case 2:
			return random_gradient_bg(w, h);
		default:
			throw std::runtime_error("Should not be there");
	}
}

Example generate_synthetic_example(int ratio) {
	std::unique_ptr<Font> font = random_font();
	Textline textline = random_textline(font, font->size() * ratio);
	int off_left = random_int(textline.bitmap.h / 2);
	int off_top = random_int(textline.bitmap.h / 2);
	int off_right = random_int(textline.bitmap.h / 2);
	int off_bottom = random_int(textline.bitmap.h / 2);
	stb::Image<stb::RGB> img = random_bg(
		off_left + textline.bitmap.w + off_right,
		off_top + textline.bitmap.h + off_bottom
	);
	stb::RGB text_color = random_RGB(img.at(0, 0), 64, 255);
	img.blend(off_left, off_top, text_color, textline.bitmap);
	Example example;
	example.text = to_utf8(textline.text);
	example.img = std::move(img);
	return example;
}

std::vector<const TextObj*> random_text_obj_list(int ratio) {
	const auto& original_list = text_objs.at(ratio);
	std::vector<const TextObj*> result;
	for (auto& obj: original_list) {
		result.push_back(&obj);
	}
	std::shuffle(result.begin(), result.end(), e1);
	return result;
}

nlohmann::json generate_batch(int batch) {
	fs::path batch_dir = fs::path(out_dir) / std::to_string(batch);
	if (!fs::exists(batch_dir)) {
		fs::create_directories(batch_dir);
	}
	nlohmann::json batch_meta = {
		{"examples", nlohmann::json::array()}
	};
	int ratio = random_ratio();
	int mean_ratio = (all_ratios.front() + all_ratios.back()) / 2;
	int i = 0;
	int total_ratio = 0;
	std::vector<const TextObj*> text_obj_list = random_text_obj_list(ratio);
	do {
		Example example;
		if (text_obj_list.size() && random_bool()) {
			example = generate_real_example(ratio, *text_obj_list.back());
			text_obj_list.pop_back();
		} else {
			example = generate_synthetic_example(ratio);
		}
		example.img = example.img.resize(HEIGHT * ratio * 2, HEIGHT);
		example.img.write_png(batch_dir / (std::to_string(i) + ".png"));
		batch_meta["examples"].push_back({
			{"img_path", std::to_string(batch) + "/" + std::to_string(i) + ".png"},
			{"text", example.text}
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
		load_docs(true);
		load_text_objs();
		run_batch_generation(generate_batch);

		return 0;
	} catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
		return 1;
	}
}
