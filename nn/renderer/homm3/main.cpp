
#include <Random.hpp>
#include <clipp.h>

struct Example {
	stb::Image<stb::RGB> img;
	nlohmann::json meta;

	bool is_overlap_with_some_obj(const Rect& rect) const {
		for (auto& obj: meta.at("objs")) {
			Rect bbox = obj;
			if ((bbox & rect).area()) {
				return true;
			}
		}
		return false;
	}

	void draw_rects() {
		for (auto& obj: meta.at("objs")) {
			stb::RGB rgb = {};
			if (obj.value("ignore_while_training", false)) {
				rgb.r = 255;
			} else {
				rgb.b = 255;
			}
			Rect bbox = Rect(obj) & Rect(img);
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
};

Rect random_empty_space(const Example& example, int w, int h) {
	for (size_t i = 0; i < 100; ++i) {
		Rect rect;
		rect.x = random_int(example.img.w - w);
		rect.y = random_int(example.img.h - h);
		rect.w = w;
		rect.h = h;
		if (example.is_overlap_with_some_obj(rect)) {
			continue;
		}
		return rect;
	}
	return {};
}

void blend(stb::Image<stb::RGB>& img, const stb::Image<stb::RGBA>& sub_img, int x_off, int y_off) {
	for (int y = 0; y < sub_img.h; ++y) {
		for (int x = 0; x < sub_img.w; ++x) {
			img.at(x_off + x, y_off + y).blend(sub_img.at(x, y));
		}
	}
}

std::string str(int i) {
	std::stringstream ss;
	ss << std::setw(2) << std::setfill('0') << i;
	return ss.str();
}

void render_random_hero(Example& example) {
	int a = random_int(18);
	int b = random_int(10);
	int c = 0;
	if (b >= 5) {
		c = random_int(8);
	}
	stb::Image<stb::RGBA> hero_img(fs::path(in_dir)
		/ ".." / "assets" / "homm3" / "hero"
		/ ("ah" + str(a) + "_.dir")
		/ (str(b) + "_" + str(c) + ".png")
	);
	int d = random_int(8);
	int e = random_int(8);
	stb::Image<stb::RGBA> flag_img(fs::path(in_dir)
		/ ".." / "assets" / "homm3" / "flag"
		/ ("af" + str(d) + ".dir")
		/ (str(b) + "_" + str(e) + ".png")
	);
	Rect rect = random_empty_space(example, hero_img.w, hero_img.h);
	if (!rect.area()) {
		return;
	}
	blend(example.img, hero_img, rect.x, rect.y);
	blend(example.img, flag_img, rect.x, rect.y);

	Rect hero_bbox = Rect::get_visible_bbox(hero_img);
	hero_bbox.x += rect.x;
	hero_bbox.y += rect.y;

	nlohmann::json hero = hero_bbox;
	hero["tag"] = "hero";

	example.meta.at("objs").push_back(hero);
}

void render_random_town(Example& example);

void render_random_object(Example& example) {
	switch (random_int(2)) {
		case 0:
			render_random_hero(example);
			break;
		case 1:
			render_random_town(example);
			break;
		default:
			throw std::runtime_error("Should not be there");
	}
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
		{"objs", nlohmann::json::array()}
	};
	for (auto& obj: src.meta.at("objs")) {
		if (obj.at("type") != "tag") {
			continue;
		}
		Rect bbox = Rect(obj) & crop;
		if (!bbox.area()) {
			continue;
		}
		bbox.x -= crop.x;
		bbox.y -= crop.y;
		nlohmann::json new_obj = bbox;
		std::u32string tag = to_utf32(obj.at("tag"));

		new_obj["ignore_while_training"] =
			(tag.size() == 0) ||
			((bbox.area() * 2) < Rect(obj).area());

		new_obj["tag"] = tag;

		dst.meta.at("objs").push_back(new_obj);
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
			render_random_object(example);
		}
		// random_channels_shuffle(example.img);
		// random_inverse(example.img);
		example.draw_rects();
		example.img.write_png(batch_dir / (std::to_string(i) + ".png"));
		batch_meta["examples"].push_back({
			{"img_path", std::to_string(batch) + "/" + std::to_string(i) + ".png"},
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

		load_docs(true);
		run_batch_generation(generate_batch);

		return 0;
	} catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
		return 1;
	}
}
