
#include <spdlog/spdlog.h>

#include "MessageHandler.hpp"
#include "Messages.hpp"

#include "../js/Runtime.hpp"
#include "../js/Tensor.hpp"
#include "../js/Point.hpp"

#include "../nn/TextTensor.hpp"
#include "../nn/ImgTensor.hpp"

using namespace std::chrono_literals;

template <typename Duration>
std::string duration_to_str(Duration duration) {

	auto s = std::chrono::duration_cast<std::chrono::seconds>(duration);
	duration -= s;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
	auto result = fmt::format("{}s:{}ms", s.count(), ms.count());

	return result;
}

void MessageHandler::run() {
	nlohmann::json request;
	while (true) {
		try {
			request = channel->recv();
			handle_request(request);
		} catch (const nn::ContinueError& continue_error) {
			channel->send(create_continue_error_message(continue_error.what()));
		} catch (const std::exception& error) {
			if (request.count("image")) {
				request["image"] = "omitted";
			}
			spdlog::error("Error while processing request \n{}:\n", request.dump(4), error.what());
			channel->send(create_error_message(error.what()));
		}
	}
}

void MessageHandler::handle_request(nlohmann::json& request) {

	//spdlog::trace(fmt::format("Got the request \n{}\n", request.dump(4)));
	auto start_timestamp = std::chrono::system_clock::now();
	nlohmann::json response;

	auto type = request.at("type").get<std::string>();
	// We can get errors
	if (type == "error") {
		//Do nothing
		return;
	} else if (type == "js_eval") {
		response = handle_js_eval_request(request);
	} else {
		throw std::runtime_error("Unexpected request type: " + type);
	}

	auto duration = std::chrono::system_clock::now() - start_timestamp;
	spdlog::trace(fmt::format("Got the response in {}: \n{}", duration_to_str(duration), response.dump(4)));
	channel->send(response);
}

nlohmann::json MessageHandler::handle_js_eval_request(nlohmann::json& request) {
	stb::Image<stb::RGB> screenshot = get_image(request);
	request["image"] = "omitted";

	spdlog::trace(fmt::format("Got a js_eval request \n{}\n", request.dump(4)));
	js::Context js_ctx(&screenshot, channel);
	auto script = fmt::format("function __testo__() {{\n{}\n}}\nlet result = __testo__()\nJSON.stringify(result)", request.at("js_script").get<std::string>());
	spdlog::trace("Executing script \n{}", script);

	nlohmann::json result;
	
	auto val = js_ctx.eval(script);
	if (val.is_string()) {
		return nlohmann::json({
			{"type", "eval_result"},
			{"data", nlohmann::json::parse(std::string(val))}
			
		});
	}
	if (val.is_undefined()) {
		throw std::runtime_error("JS script returned an undefined value");
	}
	
	return result;
}