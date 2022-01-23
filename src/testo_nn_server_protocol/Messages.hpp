#pragma once

#include <nlohmann/json.hpp>
#include <stb/Image.hpp>

#include <string>

#define REF_IMAGE_REQUEST "ref_image_request"
#define REF_IMAGE_RESPONSE "ref_image"

#define JS_EVAL_REQUEST "js_eval"
#define JS_EVAL_RESPONSE "eval_result"

#define JS_VALIDATE_REQUEST "js_validate"
#define JS_VALIDATE_RESPONSE "validation_result"

#define ERROR_RESPONSE "error"
#define CONTINUE_ERROR_RESPONSE "continue_error"

// ref image

inline nlohmann::json create_ref_image_request(const std::string& img_path) {
	return {
		{"type", REF_IMAGE_REQUEST},
		{"data", img_path},
	};
}

inline nlohmann::json create_ref_image_response(const stb::Image<stb::RGB>& ref_image) {
	return {
		{"type", REF_IMAGE_RESPONSE},
		{"image", nlohmann::json::binary(ref_image.write_png_mem())},
	};
}

// js eval

inline nlohmann::json create_js_eval_request(const stb::Image<stb::RGB>& screenshot, const std::string& script) {
	return {
		{"type", JS_EVAL_REQUEST},
		{"image", nlohmann::json::binary(screenshot.write_png_mem())},
		{"js_script", script},
	};
}

inline nlohmann::json create_js_eval_response(const nlohmann::json& data, const std::string& _stdout) {
	return {
		{"type", JS_EVAL_RESPONSE},
		{"data", data},
		{"stdout", _stdout},
	};
}

// js validate

inline nlohmann::json create_js_validate_request(const std::string& script) {
	return {
		{"type", JS_VALIDATE_REQUEST},
		{"js_script", script},
	};
}

inline nlohmann::json create_js_validate_response(bool success, const std::string& error_msg = {}) {
	return {
		{"type", JS_VALIDATE_RESPONSE},
		{"result", success},
		{"data", error_msg},
	};
}

// errors

inline nlohmann::json create_error_response(const std::string& message, const std::string& _stdout = {}) {
	return {
		{"type", ERROR_RESPONSE},
		{"data", message},
		{"stdout", _stdout},
	};
}

inline nlohmann::json create_continue_error_response(const std::string& message, const std::string& _stdout = {}) {
	return {
		{"type", CONTINUE_ERROR_RESPONSE},
		{"data", message},
		{"stdout", _stdout},
	};
}

// helpers

inline stb::Image<stb::RGB> get_image(const nlohmann::json& json) {
	if (!json.count("image")) {
		throw std::runtime_error("Message doesn't have image field");
	}

	stb::Image<stb::RGB> result(json.at("image").get_binary().data(), json.at("image").get_binary().size());
	return result;
}
