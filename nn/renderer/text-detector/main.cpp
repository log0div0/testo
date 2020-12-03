
#include <Random.hpp>
#include <clipp.h>

struct Example {
	stb::Image<stb::RGB> img;
	nlohmann::json meta;
	stb::Image<uint8_t> posi_up;
	stb::Image<uint8_t> posi_down;
	stb::Image<uint8_t> nega_up;
	stb::Image<uint8_t> nega_down;

	void draw_rects() {
		for (auto& textline: meta.at("textlines")) {
			stb::RGB rgb = {};
			if (textline.value("ignore_while_training", false)) {
				rgb.r = 255;
			} else {
				rgb.b = 255;
			}
			Rect bbox = Rect(textline) & Rect(img);
			for (int y = bbox.y; y < bbox.end_y(); ++y) {
				img.at(bbox.x, y) = rgb;
				img.at(bbox.end_x() - 1, y) = rgb;
			}
			for (int x = bbox.x; x < bbox.end_x(); ++x) {
				img.at(x, bbox.y) = rgb;
				img.at(x, bbox.end_y() - 1) = rgb;
			}
		}
	}

	static void fill_rect(stb::Image<uint8_t>& img, const Rect& rect, uint8_t val) {
		for (int y = 0; y < rect.h; ++y) {
			for (int x = 0; x < rect.w; ++x) {
				img.at(rect.x + x, rect.y + y) = val;
			}
		}
	}

	void draw_labels() {
		posi_up = stb::Image<uint8_t>(img.w, img.h, 0);
		posi_down = stb::Image<uint8_t>(img.w, img.h, 0);
		nega_up = stb::Image<uint8_t>(img.w, img.h, 255);
		nega_down = stb::Image<uint8_t>(img.w, img.h, 255);
		for (auto& textline: meta.at("textlines")) {
			Rect rect = textline;
			Rect rect_up = rect;
			Rect rect_down = rect;
			rect_up.shrink_bottom(rect.h / 3 * 2);
			rect_down.shrink_top(rect.h / 3 * 2);
			if (!textline.value("ignore_while_training", false)) {
				fill_rect(posi_up, rect_up, 255);
				fill_rect(posi_down, rect_down, 255);
			}
			rect_up.extend(rect_up.h / 2);
			rect_down.extend(rect_down.h / 2);
			rect_up = rect_up & Rect(img);
			rect_down = rect_down & Rect(img);
			fill_rect(nega_up, rect_up, 0);
			fill_rect(nega_down, rect_down, 0);
			if (textline.value("ignore_while_training", false)) {
				fill_rect(nega_up, rect_up | rect_down, 0);
				fill_rect(nega_down, rect_up | rect_down, 0);
			}
		}
	}

	bool is_overlap_with_some_textline(const Rect& rect) const {
		for (auto& textline: meta.at("textlines")) {
			Rect bbox = textline;
			if ((bbox & rect).area()) {
				return true;
			}
		}
		return false;
	}
};

Rect random_empty_space_for_text(const Example& example, int font_height) {
	const stb::Image<stb::RGB>& img = example.img;

	for (size_t i = 0; i < 100; ++i) {
		int x_pad = font_height;
		int y_pad = font_height / 2;

		int w = font_height + x_pad * 2;
		int h = font_height + y_pad * 2;

		Rect rect;
		rect.x = random_int(img.w);
		rect.y = random_int(img.h);
		rect.w = 1;
		rect.h = 1;
		stb::RGB color = img.at(rect.x, rect.y);
		bool left = true;
		bool up = true;
		bool right = true;
		bool down = true;
		while (left || up || right || down) {
			// left
			if (left) {
				left = rect.extend_left(img, color);
			}
			// up
			if (rect.h == h) {
				up = false;
			}
			if (up) {
				up = rect.extend_top(img, color);
			}
			// right
			if (right) {
				right = rect.extend_right(img, color);
			}
			// down
			if (rect.h == h) {
				down = false;
			}
			if (down) {
				down = rect.extend_bottom(img, color);
			}
		}
		if (rect.w < w) {
			continue;
		}
		if (rect.h != h) {
			continue;
		}
		rect.x += x_pad;
		rect.y += y_pad;
		rect.w -= x_pad * 2;
		rect.h -= y_pad * 2;
		if (example.is_overlap_with_some_textline(rect)) {
			continue;
		}
		return rect;
	}
	return {};
}

void render_random_text(Example& example) {
	stb::Image<stb::RGB>& img = example.img;

	std::unique_ptr<Font> font = random_font();

	Rect text_rect = random_empty_space_for_text(example, font->size());

	if (!text_rect.area()) {
		return;
	}

	stb::RGB bg_color = img.at(text_rect.x, text_rect.y);
	stb::RGB text_color = random_RGB(bg_color, 64, 255);

	Textline textline = random_textline(font, text_rect.w);

	if (textline.text.size() < 2) {
		return;
	}

	img.blend(text_rect.x, text_rect.y, text_color, textline.bitmap);

	Rect bbox = {};
	bbox.x = -1 + text_rect.x;
	bbox.y = -1 + text_rect.y;
	bbox.w = 2 + textline.bitmap.w;
	bbox.h = 2 + textline.bitmap.h;

	example.meta.at("textlines").push_back(bbox);
}

Example random_crop(const Doc& src) {
	Example dst;
	dst.img = stb::Image<stb::RGB>(800, 400, random_RGB());
	Rect crop(dst.img);
	if (src.img.w > dst.img.w) {
		crop.x = random_int(-(dst.img.w / 3), src.img.w - dst.img.w + (dst.img.w / 3));
	} else {
		crop.x = random_int(-(dst.img.w - src.img.w), 0);
	}
	if (src.img.h > dst.img.h) {
		crop.y = random_int(-(dst.img.h / 3), src.img.h - dst.img.h + (dst.img.h / 3));
	} else {
		crop.y = random_int(-(dst.img.h - src.img.h), 0);
	}
	for (int y = 0; y < crop.h; ++y) {
		for (int x = 0; x < crop.w; ++x) {
			int pos_x = crop.x + x;
			int pos_y = crop.y + y;
			if ((pos_x < 0) || (pos_x >= src.img.w)) {
				continue;
			}
			if ((pos_y < 0) || (pos_y >= src.img.h)) {
				continue;
			}
			dst.img.at(x, y) = src.img.at(pos_x, pos_y);
		}
	}

	dst.meta = {
		{"textlines", nlohmann::json::array()}
	};
	for (auto& obj: src.meta.at("objs")) {
		if (obj.at("type") != "text") {
			continue;
		}
		Rect bbox = Rect(obj) & crop;
		if (!bbox.area()) {
			continue;
		}
		bbox.x -= crop.x;
		bbox.y -= crop.y;
		nlohmann::json textline = bbox;
		std::string text = obj.at("text");

		textline["ignore_while_training"] =
			(text.size() < 2) ||
			(bbox.w <= bbox.h) ||
			(bbox.h != Rect(obj).h) ||
			(bbox.h <= 10);

		dst.meta.at("textlines").push_back(textline);
	}

	return dst;
}

nlohmann::json generate_batch(int batch) {
	fs::path batch_dir = fs::path(out_dir) / std::to_string(batch);
	if (!fs::exists(batch_dir)) {
		fs::create_directories(batch_dir);
	}
	nlohmann::json batch_meta = {
		{"examples", nlohmann::json::array()}
	};
	for (int i = 0; i < batch_size; ++i) {
		Example example = random_crop(random_doc());
		for (int i = 0; i < 10; ++i) {
			render_random_text(example);
		}
		// example.draw_rects();
		example.draw_labels();
		random_channels_shuffle(example.img);
		random_inverse(example.img);
		example.img.write_png(batch_dir / (std::to_string(i) + ".png"));
		example.posi_up.write_png(batch_dir / (std::to_string(i) + "_posi_up.png"));
		example.posi_down.write_png(batch_dir / (std::to_string(i) + "_posi_down.png"));
		example.nega_up.write_png(batch_dir / (std::to_string(i) + "_nega_up.png"));
		example.nega_down.write_png(batch_dir / (std::to_string(i) + "_nega_down.png"));
		batch_meta["examples"].push_back({
			{"img_path", std::to_string(batch) + "/" + std::to_string(i) + ".png"},
			{"posi_up_path", std::to_string(batch) + "/" + std::to_string(i) + "_posi_up.png"},
			{"posi_down_path", std::to_string(batch) + "/" + std::to_string(i) + "_posi_down.png"},
			{"nega_up_path", std::to_string(batch) + "/" + std::to_string(i) + "_nega_up.png"},
			{"nega_down_path", std::to_string(batch) + "/" + std::to_string(i) + "_nega_down.png"},
		});
	}
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
		run_batch_generation(generate_batch);

		return 0;
	} catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
		return 1;
	}
}
