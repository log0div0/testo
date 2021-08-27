#pragma once

#include <nlohmann/json.hpp>
#include <stb/Image.hpp>

#include <string>

inline void check_for_error(const nlohmann::json& json) {
	if (json["type"].get<std::string>() == "error") {
		throw std::runtime_error(json.at("data").get<std::string>());
	}
}

inline nlohmann::json create_ref_image_message(const stb::Image<stb::RGB>& ref_image) {
	return {
		{"version", NN_SERVICE_PROCOTOL_VERSION},
		{"type", "ref_image"},
		{"image", nlohmann::json::binary(ref_image.write_png_mem())}
	};
}

inline nlohmann::json create_ref_image_request(const std::string& img_path) {
	return {
		{"version", NN_SERVICE_PROCOTOL_VERSION},
		{"type", "ref_image_request"},
		{"data", img_path}
	};
}

inline nlohmann::json create_js_eval_request(const stb::Image<stb::RGB>& screenshot, const std::string& script) {
	return {
		{"version", NN_SERVICE_PROCOTOL_VERSION},
		{"type", "js_eval"},
		{"image", nlohmann::json::binary(screenshot.write_png_mem())},
		{"js_script", script}
	};
}

inline nlohmann::json create_js_validate_request(const std::string& script) {
	return {
		{"version", NN_SERVICE_PROCOTOL_VERSION},
		{"type", "js_validate"},
		{"js_script", script}
	};
}

inline nlohmann::json create_error_message(const std::string& message, std::string _stdout = "") {
	return {
		{"version", NN_SERVICE_PROCOTOL_VERSION},
		{"type", "error"},
		{"data", message},
		{"stdout", _stdout}
	};
}

inline nlohmann::json create_continue_error_message(const std::string& message, std::string _stdout = "") {
	return {
		{"version", NN_SERVICE_PROCOTOL_VERSION},
		{"type", "continue_error"},
		{"data", message},
		{"stdout", _stdout}
	};
}

inline stb::Image<stb::RGB> get_image(const nlohmann::json& json) {
	if (!json.count("image")) {
		throw std::runtime_error("Message doesn't have image field");
	}

	stb::Image<stb::RGB> result(json.at("image").get_binary().data(), json.at("image").get_binary().size());
	return result;
}
