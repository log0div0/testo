
#include <chrono>
#include <iostream>
#include <clipp.h>
#include <fstream>
#include <codecvt>
#include <set>
#include <nlohmann/json.hpp>
#include "TextTensor.hpp"
#include "ImgTensor.hpp"
#include "TextRecognizer.hpp"
#include "OnnxRuntime.hpp"

void draw_rect(stb::Image<stb::RGB>& img, nn::Rect bbox, stb::RGB color) {
	for (int y = bbox.top; y <= bbox.bottom; ++y) {
		img.at(bbox.left, y) = color;
		img.at(bbox.right, y) = color;
	}
	for (int x = bbox.left; x < bbox.right; ++x) {
		img.at(x, bbox.top) = color;
		img.at(x, bbox.bottom) = color;
	}
}

std::map<char32_t, size_t> symbols_map;

size_t find_substr(const std::u32string& str, const std::u32string& substr, size_t pos) {
	if (!symbols_map.size()) {
		for (size_t symbol_index = 0; symbol_index < nn::TextRecognizer::symbols.size(); ++symbol_index) {
			auto& symbol = nn::TextRecognizer::symbols.at(symbol_index);
			for (char32_t codepoint: symbol) {
				symbols_map[codepoint] = symbol_index;
			}
		}
	}
	while ((pos + substr.size()) <= str.size()) {
		size_t i = 0;
		for (; i < substr.size(); ++i) {
			size_t a = symbols_map.at(substr[i]);
			size_t b = symbols_map.at(str[pos+i]);
			if (a != b) {
				break;
			}
		}
		if (i == substr.size()) {
			return pos;
		}
		++pos;
	}
	return std::string::npos;
}

std::u32string remove_spaces(const std::u32string str) {
	std::u32string result;
	for (char32_t cp: str) {
		if (cp != U' ') {
			result.push_back(cp);
		}
	}
	return result;
}

std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;

struct TextArgs {
	std::string img_file;
	std::string query, fg, bg;
};

void text_mode(const TextArgs& args)
{
	stb::Image<stb::RGB> image(args.img_file);

	auto start = std::chrono::high_resolution_clock::now();
	nn::TextTensor tensor = nn::find_text(&image);
	if (args.query.size()) {
		tensor = tensor.match_text(&image, args.query);
	}
	if (args.fg.size() || args.bg.size()) {
		tensor = tensor.match_color(&image, args.fg, args.bg);
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end - start;
	std::cout << "Time: " << time.count() << " seconds" << std::endl;

	std::cout << "Found: " << tensor.size() << std::endl;

	for (auto& textline: tensor.objects) {
		draw_rect(image, textline.rect, {200, 20, 50});
	}

	image.write_png("output.png");
}

struct ImgArgs {
	std::string search_img_file;
	std::string ref_img_file;
};

void img_mode(const ImgArgs& args)
{
	stb::Image<stb::RGB> image(args.search_img_file);

	auto start = std::chrono::high_resolution_clock::now();
	nn::ImgTensor tensor = nn::find_img(&image, args.ref_img_file);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end - start;
	std::cout << "Time: " << time.count() << " seconds" << std::endl;

	std::cout << "Found: " << tensor.size() << std::endl;

	for (auto& img: tensor.objects) {
		draw_rect(image, img.rect, {200, 20, 50});
	}

	image.write_png("output.png");
}

struct DatasetArgs {
	std::string dataset_path;
};

nlohmann::json load_json(const std::string& path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file " + path);
	}
	return nlohmann::json::parse(file);
}

struct DatasetMode {
	DatasetMode(const DatasetArgs& args_): args(args_) {

	}

	void run() {
		std::vector<fs::path> meta_paths;
		for (const auto& entry: fs::directory_iterator(args.dataset_path)) {
			std::string ext = entry.path().extension();
			if (ext != ".json") {
				continue;
			}
			fs::path meta_path = entry.path();
			meta_paths.push_back(meta_path);
		}
		std::sort(meta_paths.begin(), meta_paths.end());
		for (auto& meta_path: meta_paths) {
			run_doc(meta_path);
		}

		std::cout
			<< "Total: " << (success_counter + fail_counter)
			<< ", Success: " << success_counter
			<< ", Fail: " << fail_counter
			<< ", Accuracy: " << (float(success_counter) / float(success_counter + fail_counter))
			<< std::endl;
	}

	bool filter_text_obj(const nlohmann::json& obj) {
		std::string text = obj.at("text");
		std::u32string u32text = conv.from_bytes(text);
		if (u32text.size() < 2) {
			return false;
		}
		float h = obj.at("height");
		if (h <= 10) {
			return false;
		}
		return true;
	}

	void run_doc(fs::path meta_path) {
		std::cout << meta_path.stem();
		std::cout.flush();

		fs::path img_path = meta_path;
		img_path.replace_extension(".png");

		stb::Image<stb::RGB> img(img_path.string());
		if (img.w > 1920 || img.h > 1080) {
			std::cout << " skipping (" << img.w << "x" << img.h << ")" << std::endl;
			return;
		}
		nlohmann::json meta = load_json(meta_path);

		std::cout << std::endl;

		nn::TextTensor text_tensor = nn::find_text(&img);
		std::set<std::u32string> visited_textlines;
		for (auto& obj: meta.at("objs")) {
			if (obj.at("type") == "text") {
				if (!filter_text_obj(obj)) {
					continue;
				}
				std::string text = obj.at("text");
				std::u32string u32text = conv.from_bytes(text);
				auto res = visited_textlines.insert(u32text);
				if (!res.second) {
					continue;
				}
				size_t detected = text_tensor.match_text(&img, text).size();
				size_t labeled = get_text_match_count(meta, u32text);
				if (detected != labeled) {
					std::cout << "labeled: " << labeled << ", detected:" << detected << ", text: " << text << std::endl;
					fail_counter++;
				} else {
					success_counter++;
				}
			} else {
				throw std::runtime_error("Unknown type of object");
			}
		}
	}

	size_t get_text_match_count(const nlohmann::json& meta, std::u32string substr) {
		size_t count = 0;
		for (auto& obj: meta.at("objs")) {
			if (obj.at("type") == "text") {
				if (!filter_text_obj(obj)) {
					continue;
				}
				std::string s_ = obj.at("text");
				std::u32string s = remove_spaces(conv.from_bytes(s_));
				substr = remove_spaces(substr);
				size_t nPos = find_substr(s, substr, 0);
				while (nPos != std::string::npos)
				{
					count++;
					nPos = find_substr(s, substr, nPos + substr.size());
				}
			}
		}
		return count;
	}

	size_t fail_counter = 0;
	size_t success_counter = 0;
	DatasetArgs args;
};

enum class mode {
	text,
	img,
	dataset
};

int main(int argc, char **argv)
{
	try {
		using namespace clipp;

		mode selected_mode;

		TextArgs text_args;
		auto text_spec = (
			command("text").set(selected_mode, mode::text),
			value("input image", text_args.img_file),
			option("--query") & value("the text to search for", text_args.query),
			option("--fg") & value("foreground color", text_args.fg),
			option("--bg") & value("background color", text_args.bg)
		);

		ImgArgs img_args;
		auto img_spec = (
			command("img").set(selected_mode, mode::img),
			value("search image", img_args.search_img_file),
			value("ref image", img_args.ref_img_file)
		);

		DatasetArgs dataset_args;
		auto dataset_spec = (
			command("dataset").set(selected_mode, mode::dataset),
			value("dataset path", dataset_args.dataset_path)
		);

		auto cli = (text_spec | img_spec | dataset_spec);

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			return 1;
		}

		nn::onnx::Runtime onnx_runtime;
		switch (selected_mode) {
			case mode::text:
				text_mode(text_args);
				break;
			case mode::img:
				img_mode(img_args);
				break;
			case mode::dataset:
				DatasetMode(dataset_args).run();
				break;
			default:
				throw std::runtime_error("Invalid mode");
		}
	}
	catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
		return 1;
	}
	return 0;
}
