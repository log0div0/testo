
#pragma once

#include <random>
#include <codecvt>
#include <locale>
#include <Font.hpp>
#include <nlohmann/json.hpp>
#include <thread>
#include <mutex>
#include <iostream>
#include <fstream>
#include <climits>

#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;

std::string in_dir;
std::string out_dir;
int batch_count = 0;
int batch_size = 0;

std::random_device r;
std::default_random_engine e1(r());

int random_int(int count) {
	std::uniform_int_distribution<int> uniform_dist(0, count - 1);
	return uniform_dist(e1);
}

int random_int(int min, int max) {
	std::uniform_int_distribution<int> uniform_dist(min, max);
	return uniform_dist(e1);
}

float random_float(float min, float max) {
	std::uniform_real_distribution<float> uniform_dist(min, max);
	return uniform_dist(e1);
}

bool random_bool(float true_prob = 1.0f / 2) {
	if (true_prob < 0) {
		throw std::runtime_error("true_prob < 0");
	}
	if (true_prob >= 1) {
		throw std::runtime_error("true_prob >= 1");
	}
	return random_float(0.0f , 1.0f) < true_prob;
}

std::string english = "abcdefghijklmnopqrstuvwxyz";
std::string ENGLISH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
std::string russian = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя";
std::string RUSSIAN = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ";
std::string alphabet = "0123456789!?\"'#$%&@()[]{}<>+-*/\\.,:;^~=|_" + english + ENGLISH + russian + RUSSIAN;
std::string space = " ";

std::string to_utf8(const std::u32string &s) {
	std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
	return conv.to_bytes(s);
}

std::u32string to_utf32(const std::string &s) {
	std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
	return conv.from_bytes(s);
}

std::u32string u32alphabet = to_utf32(alphabet);
std::u32string u32space = to_utf32(space);

char32_t random_codepoint() {
	return u32alphabet.at(random_int(u32alphabet.size()));
}

char32_t space_codepoint() {
	return u32space.at(0);
}

stb::RGB random_RGB() {
	stb::RGB color;
	color.r = random_int(256);
	color.g = random_int(256);
	color.b = random_int(256);
	return color;
}

stb::RGB random_RGB(const stb::RGB& other_color, int min_diff, int max_diff) {
	stb::RGB color;
	for (int c = 0; c < 3; ++c) {
		while (true) {
			color[c] = random_int(256);
			int diff = std::abs(int(color[c]) - int(other_color[c]));
			if ((diff >= min_diff) && (diff <= max_diff)) {
				break;
			}
		}
	}
	return color;
}

void random_channels_shuffle(stb::Image<stb::RGB>& img) {
	std::array<uint8_t, 3> map = {0, 1, 2};
	std::random_shuffle(map.begin(), map.end());
	for (int y = 0; y < img.h; ++y) {
		for (int x = 0; x < img.w; ++x) {
			stb::RGB pixel;
			for (int c = 0; c < img.c; ++c) {
				pixel[map[c]] = img.at(x, y)[c];
			}
			img.at(x, y) = pixel;
		}
	}
}

void random_inverse(stb::Image<stb::RGB>& img) {
	if (random_bool()) {
		return;
	}
	for (int y = 0; y < img.h; ++y) {
		for (int x = 0; x < img.w; ++x) {
			for (int c = 0; c < img.c; ++c) {
				img.at(x, y)[c] = 255 - img.at(x, y)[c];
			}
		}
	}
}

struct Doc {
	stb::Image<stb::RGB> img;
	nlohmann::json meta;
};

nlohmann::json load_json(const std::string& path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file " + path);
	}
	return nlohmann::json::parse(file);
}

void save_json(const std::string& path, const nlohmann::json& j) {
	std::ofstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file " + path);
	}
	file << std::setw(4) << j << std::endl;
}

std::vector<stb::Font> tt_fonts;
std::vector<psf::Font> psf_fonts;
std::vector<std::string> icon_paths;
std::vector<Doc> docs;

const stb::Font& _random_tt_font() {
	return tt_fonts.at(random_int(tt_fonts.size()));
}

const psf::Font& _random_psf_font() {
	return psf_fonts.at(random_int(psf_fonts.size()));
}

std::unique_ptr<Font> random_font() {
	if (random_int(tt_fonts.size() + psf_fonts.size()) < (int)tt_fonts.size()) {
		bool use_antialiasing = random_bool();
		int font_size = random_int(use_antialiasing ? 13 : 14, 48);
		stb::ScaledFont font = _random_tt_font().scaleForPixelHeight(font_size);
		float letter_spacing = random_float(float(-font_size) / 10, 0);
		return std::unique_ptr<Font>(new TTFont(font, use_antialiasing, letter_spacing));
	} else {
		return std::unique_ptr<Font>(new PSFFont(_random_psf_font()));
	}
}

struct Textline {
	stb::Image<stb::RGB> bitmap;
	std::u32string text;
};

Textline random_textline(const std::unique_ptr<Font>& font, int max_text_bbox_width) {
	float x = 0;
	float space_width = font->size() / 2;
	std::vector<Char> chars;

	while (x < max_text_bbox_width) {
		char32_t codepoint = 0;
		if (chars.size()) {
			if (chars.back().codepoint == U'-') {
				codepoint = random_bool(1.0f / 2) ? U'-' : random_codepoint();
			} else if (chars.back().codepoint == U'.') {
				codepoint = random_bool(1.0f / 2) ? U'.' : random_codepoint();
			} else if (chars.back().codepoint == U' ') {
				codepoint = random_codepoint();
			} else if (chars.size() <= 2) {
				codepoint = random_codepoint();
			} else {
				codepoint = random_bool(1.0f / 5) ? U' ' : random_codepoint();
			}
		} else {
			codepoint = random_codepoint();
		}
		if (codepoint == space_codepoint()) {
			chars.push_back(space_codepoint());
			x += space_width;
		} else {
			ScaledHMetrics hmetrics = font->codepointHMetrics(codepoint);
			if (chars.size()) {
				x += font->codepointKernAdvance(chars.back().codepoint, codepoint);
			}
			chars.push_back(font->renderCodepoint(codepoint, x));
			x += hmetrics.advanceWidth;
		}
	}

	while (chars.size() && ((chars.back().codepoint == space_codepoint()) || (chars.back().end_x() > max_text_bbox_width))) {
		chars.pop_back();
	}

	if (chars.size() == 0) {
		return {};
	}

	Textline textline;

	int min_x = INT_MAX;
	int min_y = INT_MAX;
	int max_x = INT_MIN;
	int max_y = INT_MIN;

	for (auto& ch: chars) {
		textline.text.push_back(ch.codepoint);
		if (ch.codepoint == space_codepoint()) {
			continue;
		}
		if (min_x > ch.x) {
			min_x = ch.x;
		}
		if (min_y > ch.y) {
			min_y = ch.y;
		}
		if (max_x < ch.end_x()) {
			max_x = ch.end_x();
		}
		if (max_y < ch.end_y()) {
			max_y = ch.end_y();
		}
	}

	textline.bitmap = stb::Image<stb::RGB>(max_x - min_x, max_y - min_y, stb::RGB::black());

	for (auto& ch: chars) {
		if (ch.codepoint == space_codepoint()) {
			continue;
		}
		textline.bitmap.blend(ch.x - min_x, ch.y - min_y, stb::RGB::white(), ch.bitmap);
	}

	return textline;
}

const std::string& random_icon_path() {
	return icon_paths.at(random_int(icon_paths.size()));
}

const Doc& random_doc() {
	return docs.at(random_int(docs.size()));
}

void _load_tt_fonts() {
	fs::path dir = fs::path(in_dir) / ".." / "assets" / "fonts" / "truetype";
	std::cout << "loading tt fonts ... ";
	std::cout.flush();
	for (auto& entry: fs::recursive_directory_iterator(dir)) {
		if (entry.path().extension() == ".ttf") {
			tt_fonts.push_back(stb::Font(entry.path()));
		}
	}
	std::cout << tt_fonts.size() << std::endl;
}

void _load_psf_fonts() {
	fs::path dir = fs::path(in_dir) / ".." / "assets" / "fonts" / "console";
	std::cout << "loading psf fonts ... ";
	std::cout.flush();
	for (auto& entry: fs::recursive_directory_iterator(dir)) {
		if (entry.path().extension() == ".gz") {
			psf_fonts.push_back(psf::Font(entry.path()));
		}
	}
	std::cout << psf_fonts.size() << std::endl;
}

void load_fonts() {
	_load_tt_fonts();
	_load_psf_fonts();
}

void load_icon_paths() {
	fs::path dir = fs::path(in_dir) / ".." / "assets" / "icons";
	std::cout << "loading icon paths ... ";
	std::cout.flush();
	for (const auto& entry: fs::recursive_directory_iterator(dir)) {
		std::string ext = entry.path().extension();
		if (ext != ".png") {
			continue;
		}
		fs::path icon_path = entry.path();
		icon_paths.push_back(icon_path);
	}
	std::cout << icon_paths.size() << std::endl;
}

void load_docs(bool verified_only) {
	fs::path dir = fs::path(in_dir);
	std::cout << "loading docs ... ";
	std::cout.flush();
	for (const auto& entry: fs::directory_iterator(dir)) {
		std::string ext = entry.path().extension();
		if (ext != ".png") {
			continue;
		}
		Doc doc;
		doc.img = entry.path().string();
		if (verified_only) {
			fs::path meta_path = entry.path();
			meta_path.replace_extension(".json");
			if (!fs::exists(meta_path)) {
				continue;
			}
			doc.meta = load_json(meta_path);
			if (doc.meta["verified"].get<bool>() == false) {
				continue;
			}
		}
		docs.push_back(std::move(doc));
	}
	std::cout << docs.size() << std::endl;
}

std::mutex mutex;
int batch_counter = 0;
nlohmann::json dataset_meta = {
	{"batches", nlohmann::json::array()}
};

template <typename Func>
void batch_generation_thread(Func&& func) {
	int batch = -1;
	while (true) {
		{
			std::lock_guard<std::mutex> lock(mutex);
			batch = batch_counter++;
			if (batch >= batch_count) {
				return;
			}
			std::cout << "\r" << batch << " / " << batch_count;
			std::cout.flush();
		}

		nlohmann::json batch_meta = func(batch);

		std::lock_guard<std::mutex> lock(mutex);
		dataset_meta["batches"].push_back(batch_meta);
	}
}

template <typename Func>
void run_batch_generation(Func&& func) {
	srand(time(NULL));

	int threads_count = std::thread::hardware_concurrency();
	std::cout << "threads_count = " << threads_count << std::endl;
	std::vector<std::thread> threads;
	for (int i = 0; i < threads_count; ++i) {
		threads.emplace_back([&] {
			batch_generation_thread(func);
		});
	}
	for (auto& thread: threads) {
		thread.join();
	}

	std::cout << "\r" << "writing meta ... ";
	fs::path meta_path = fs::path(out_dir) / "meta.json";
	save_json(meta_path, dataset_meta);
	std::cout << "done" << std::endl;
}
