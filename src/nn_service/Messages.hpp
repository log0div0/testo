#pragma once

#include <nlohmann/json.hpp>
#include <stb/Image.hpp>

#include <string>

struct ImageSize {
	int w, h, c;

	int total_size() const {
		return w * h * c;
	}
};

inline void to_json(nlohmann::json& j, const ImageSize& img_size) {
	j["w"] = img_size.w;
	j["h"] = img_size.h;
	j["c"] = img_size.c;
}

inline void from_json(const nlohmann::json& j, ImageSize& img_size) {
	img_size.w = j["w"];
	img_size.h = j["h"];
	img_size.c = j["c"];
}

struct Message {
	Message() = default;
	virtual ~Message() = default;
	Message(const stb::Image<stb::RGB>& screenshot): screenshot(screenshot) {
		header["version"] = NN_SERVICE_PROCOTOL_VERSION;
		ImageSize img_size = {screenshot.w, screenshot.h, screenshot.c};
		header["screenshot"] = img_size;
	}

	virtual std::string to_string() const {
		return header.dump(4);
	}

	nlohmann::json header;
	stb::Image<stb::RGB> screenshot;
};

struct TextRequest: Message {
	TextRequest() = default;
	TextRequest(const stb::Image<stb::RGB>& screenshot,
			std::string text_to_find = "",
			std::string color_fg = "",
			std::string color_bg = ""): Message(screenshot)
	{
		header["type"] = "text";
		if (text_to_find.length()) {
			header["text_to_find"] = text_to_find;
		}

		if (color_fg.length()) {
			header["color_fg"] = color_fg;
		}

		if (color_bg.length()) {
			header["color_bg"] = color_bg;
		}
	}
	
	bool has_text() const {
		return header.count("text_to_find");
	}

	std::string text() const {
		return header.value("text_to_find", std::string());
	}

	bool has_fg() const {
		return header.count("color_fg");
	}

	std::string color_fg() const {
		return header.value("color_fg", std::string());
	}

	bool has_bg() const {
		return header.count("color_bg");
	}

	std::string color_bg() const {
		return header.value("color_bg", std::string());
	}

};

struct ImgRequest: Message {
	ImgRequest() = default;
	ImgRequest(const stb::Image<stb::RGB>& screenshot, const stb::Image<stb::RGB>& pattern): Message(screenshot), pattern(pattern)
	{
		header["type"] = "img";
		ImageSize pattern_size = {pattern.w, pattern.h, pattern.c};
		header["pattern"] = pattern_size;
	}

	stb::Image<stb::RGB> pattern;
};

struct JSRequest: Message {
	JSRequest() = default;
	JSRequest(const stb::Image<stb::RGB>& screenshot, const std::string& script): Message(screenshot), script(script)
	{
		header["type"] = "js";
		header["js_size"] = script.length();
	}

	std::string to_string() const override {
		std::string result = header.dump(4);
		result += "\n";
		result += script;
		return result;
	}

	std::string script;
};

struct RefImage: Message {
	RefImage() = default;
	RefImage(const stb::Image<stb::RGB>& ref_image): Message(ref_image)
	{
		header["type"] = "ref_image";
	}
};

