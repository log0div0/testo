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

