
#include <spdlog/spdlog.h>

#include "MessageHandler.hpp"

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
	while (true) {
		handle_request(channel->receive_message());
	}
}

void MessageHandler::handle_request(std::unique_ptr<Message> request) {
	spdlog::trace(fmt::format("Got the request \n{}", request->to_string()));
	auto start_timestamp = std::chrono::system_clock::now();
	nlohmann::json response;
	if (auto p = dynamic_cast<JSRequest*>(request.get())) {
		response = handle_js_request(p);
	} else if (dynamic_cast<RefImage*>(request.get())) {
		//ignore that shit here
	}
	auto duration = std::chrono::system_clock::now() - start_timestamp;
	spdlog::trace(fmt::format("Got the response in {}: \n{}", duration_to_str(duration), response.dump(4)));
	channel->send_response(response);
}

nlohmann::json MessageHandler::handle_js_request(JSRequest* request) {
	js::Context js_ctx(&request->screenshot, channel);

	auto script = fmt::format("function __testo__() {{\n{}\n}}\nlet result = __testo__()\nJSON.stringify(result)", request->script);

	spdlog::trace("Executing script \n{}", script);

	nlohmann::json result;
	try {
		auto val = js_ctx.eval(script);
		if (val.is_string()) {
			return nlohmann::json({
				{"type", "eval_result"},
				{"data", nlohmann::json::parse(std::string(val))}
				
			});
		}
		if (val.is_undefined()) {
			return create_error_msg("JS script returned an undefined value");
		}
	} catch (const nn::ContinueError& continue_error) {
		auto msg = continue_error.what();
		return nlohmann::json({
			{"type", "continue_error"},
			{"data", msg}
		});
	} catch (const std::exception& err) {
		return create_error_msg(err.what());
	}

	return result;
}

nlohmann::json MessageHandler::create_error_msg(const std::string& message) {
	nlohmann::json error;
	error["type"] = "error";
	error["data"] = message;
	return error;
}