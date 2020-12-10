
#include <Random.hpp>
#include <clipp.h>

bool is_sub_image_match(const stb::Image<stb::RGB>& img, const stb::Image<stb::RGB>& sub, int off_x, int off_y) {
	int different_pixels_count = 0;
	int max_different_pixels_count = sub.h + sub.w;
	for (int y = 0; y < sub.h; ++y) {
		for (int x = 0; x < sub.w; ++x) {
			if (img.at(off_x + x, off_y + y).max_channel_diff(sub.at(x, y)) > 64) {
				different_pixels_count += 1;
			}
			if (different_pixels_count > max_different_pixels_count) {
				return false;
			}
		}
	}
	return true;
}

bool is_sub_image_unique(const stb::Image<stb::RGB>& img, const stb::Image<stb::RGB>& sub) {
	int end_y = img.h - sub.h + 1;
	int end_x = img.w - sub.w + 1;
	int matches = 0;
	for (int y = 0; y < end_y; ++y) {
		for (int x = 0; x < end_x; ++x) {
			if (is_sub_image_match(img, sub, x, y)) {
				matches += 1;
			}
			if (matches > 1) {
				return false;
			}
		}
	}
	return true;
}

Rect random_empty_space_for_icon(const stb::Image<stb::RGB>& img, int w, int h) {
	for (size_t i = 0; i < 100; ++i) {
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
			if (rect.w == w) {
				left = false;
			}
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
			if (rect.w == w) {
				right = false;
			}
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
		if (rect.w != w) {
			continue;
		}
		if (rect.h != h) {
			continue;
		}
		return rect;
	}
	return {};
}

void render_random_icon(stb::Image<stb::RGB>& img) {
	stb::Image<stb::RGBA> icon = random_icon_path();

	Rect icon_rect = random_empty_space_for_icon(img, icon.w, icon.h);

	if (!icon_rect.area()) {
		return;
	}

	for (int y = 0; y < icon.h; ++y) {
		for (int x = 0; x < icon.w; ++x) {
			img.at(icon_rect.x + x, icon_rect.y + y).blend(icon.at(x, y));
		}
	}
}

nlohmann::json batch_generator(int batch) {
	fs::path batch_dir = fs::path(out_dir) / std::to_string(batch);
	if (!fs::exists(batch_dir)) {
		fs::create_directories(batch_dir);
	}
	int search_img_size = 255;
	int reference_img_size = 63;
	nlohmann::json batch_meta = {
		{"search_img_size", search_img_size},
		{"reference_img_size", reference_img_size},
		{"examples", nlohmann::json::array()}
	};
	for (int i = 0; i < batch_size; ++i) {
		const stb::Image<stb::RGB>& img = random_doc().img;
		while (true) {
			int jitter = reference_img_size / 2;
			int reference_img_w = random_int(reference_img_size - jitter, reference_img_size + jitter);
			int reference_img_h = random_int(reference_img_size - jitter, reference_img_size + jitter);
			stb::Image<stb::RGB> search_img = img.sub_image(
				random_int(0, img.w - search_img_size),
				random_int(0, img.h - search_img_size),
				search_img_size,
				search_img_size
			);
			render_random_icon(search_img);
			stb::Image<stb::RGB> reference_img = search_img.sub_image(
				(search_img_size - reference_img_w) / 2,
				(search_img_size - reference_img_h) / 2,
				reference_img_w,
				reference_img_h
			);
			if (!is_sub_image_unique(search_img, reference_img)) {
				continue;
			}
			reference_img = reference_img.resize(reference_img_size, reference_img_size);
			search_img.write_png(batch_dir / (std::to_string(i) + "_search.png"));
			std::string ref_ext;
			if (random_bool()) {
				ref_ext = "png";
				reference_img.write_png(batch_dir / (std::to_string(i) + "_reference.png"));
			} else {
				ref_ext = "jpeg";
				reference_img.write_jpg(batch_dir / (std::to_string(i) + "_reference.jpeg"), random_int(80, 100));
			}
			batch_meta["examples"].push_back({
				{"search_img_path", std::to_string(batch) + "/" + std::to_string(i) + "_search.png"},
				{"reference_img_path", std::to_string(batch) + "/" + std::to_string(i) + "_reference." + ref_ext}
			});
			break;
		}
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

		load_icon_paths();
		load_docs(false);
		run_batch_generation(batch_generator);

		return 0;
	} catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
		return 1;
	}
}
