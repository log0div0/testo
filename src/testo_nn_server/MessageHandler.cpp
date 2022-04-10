
#include <spdlog/spdlog.h>

#include "MessageHandler.hpp"
#include "testo_nn_server_protocol/Messages.hpp"

#include "js/Runtime.hpp"
#include "js/Tensor.hpp"
#include "js/Point.hpp"

#include "nn/TextTensor.hpp"
#include "nn/ImgTensor.hpp"

using namespace std::chrono_literals;

static const VersionNumber server_version(TESTO_VERSION);

template <typename Duration>
std::string duration_to_str(Duration duration) {

	auto s = std::chrono::duration_cast<std::chrono::seconds>(duration);
	duration -= s;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
	auto result = fmt::format("{}s:{}ms", s.count(), ms.count());

	return result;
}

void MessageHandler::run() {
	nlohmann::json request, response;
	while (true) {
		request = channel->recv();
		try {
			response = handle_request(request);
		} catch (const std::system_error& error) {
			throw;
		} catch (const ContinueError& continue_error) {
			response = create_continue_error_response(continue_error.what());
		} catch (const std::exception& error) {
			if (request.count("image")) {
				request["image"] = "omitted";
			}
			spdlog::error("Error while processing request \n{}:\n{}", request.dump(4), error.what());
			response = create_error_response(error.what(), GetFailureCategory(error));
		}
		channel->send(response);
	}
}

stb::Image<stb::RGBA> MessageHandler::get_ref_image(const std::string& img_path)
{
	if (!channel) {
		throw std::runtime_error("Channel is nullptr");
	}

	channel->send(create_ref_image_request(img_path));

	nlohmann::json response;
	try {
		response = channel->recv();
	} catch (const std::exception& error) {
		throw std::runtime_error("Couldn't get the ref image: " + std::string(error.what()));
	}

	auto type = response.at("type").get<std::string>();
	if (type != REF_IMAGE_RESPONSE) {
		throw std::runtime_error("Unexpected message type instead of \"ref_image\": " + type);
	}

	stb::Image<stb::RGBA> ref_image = get_image_with_alpha(response);
	return ref_image;
}

nlohmann::json MessageHandler::handle_request(nlohmann::json& request) {

	auto start_timestamp = std::chrono::system_clock::now();
	nlohmann::json response;

	auto type = request.at("type").get<std::string>();
	if (type == HANDSHAKE_REQUEST) {
		response = handle_handshake(request);
	 } else if (type == JS_EVAL_REQUEST) {
		response = handle_js_eval(request);
	} else if (type == JS_VALIDATE_REQUEST) {
		response = handle_js_validate(request);
	} else {
		throw std::runtime_error("Unexpected request type: " + type);
	}

	auto duration = std::chrono::system_clock::now() - start_timestamp;
	spdlog::trace(fmt::format("Got the response in {}: \n{}", duration_to_str(duration), response.dump(4)));
	return response;
}

nlohmann::json MessageHandler::handle_handshake(nlohmann::json& request) {
	client_version = request.at("client_version").get<std::string>();
	return create_handshake_response(server_version);
}

nlohmann::json MessageHandler::handle_js_eval(nlohmann::json& request) {
	stb::Image<stb::RGB> screenshot = get_image(request);
	request["image"] = "omitted";

	spdlog::trace(fmt::format("Got a js_eval request \n{}\n", request.dump(4)));
	js::Context js_ctx(&screenshot, this);
	auto script = fmt::format("function __testo__() {{\n{}\n}}\nlet result = __testo__()\nJSON.stringify(result)", request.at("js_script").get<std::string>());
	spdlog::trace("Executing script \n{}", script);

	nlohmann::json result;
	
	auto val = js_ctx.eval(script);

	if (val.is_string()) {
		return create_js_eval_response(nlohmann::json::parse(std::string(val)), js_ctx.get_stdout().str());
	}
	
	if (val.is_undefined()) {
		throw std::runtime_error("JS script returned an undefined value");
	}
	
	return result;
}


nlohmann::json MessageHandler::handle_js_validate(nlohmann::json& request) {
	spdlog::trace(fmt::format("Got a js_validate request \n{}\n", request.dump(4)));
	js::Context js_ctx(nullptr, this);
	auto script = fmt::format("function __testo__() {{\n{}\n}}\nlet result = __testo__()\nJSON.stringify(result)", request.at("js_script").get<std::string>());
	spdlog::trace("Validating script \n{}", script);
	
	try {
		js_ctx.eval(script, true);
		return create_js_validate_response(true);
	} catch (const std::exception& error) {
		return create_js_validate_response(false, error.what());
	}
}

