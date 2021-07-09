
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
		handle_request(channel->receive_request());
	}
}

void MessageHandler::handle_request(std::unique_ptr<Request> request) {
	spdlog::trace(fmt::format("Got the request \n{}", request->to_string()));
	auto start_timestamp = std::chrono::system_clock::now();
	nlohmann::json response;
	if (auto p = dynamic_cast<TextRequest*>(request.get())) {
		response = handle_text_request(p);
	} else if (auto p = dynamic_cast<ImgRequest*>(request.get())) {
		response = handle_img_request(p);
	} else if (auto p = dynamic_cast<JSRequest*>(request.get())) {
		response = handle_js_request(p);
	}
	auto duration = std::chrono::system_clock::now() - start_timestamp;
	spdlog::trace(fmt::format("Got the response in {}: \n{}", duration_to_str(duration), response.dump(4)));
	channel->send_response(response);
}

nlohmann::json MessageHandler::handle_text_request(TextRequest* request) {
	nn::TextTensor tensor = nn::find_text(&request->screenshot);
	if (request->has_text()) {
		tensor = tensor.match_text(&request->screenshot, request->text());
	}

	if (request->has_fg() || request->has_bg()) {
		tensor = tensor.match_color(&request->screenshot, request->color_fg(), request->color_bg());
	}
	return tensor;
}

nlohmann::json MessageHandler::handle_img_request(ImgRequest* request) {
	nn::ImgTensor tensor = nn::find_img(&request->screenshot, &request->pattern);
	return tensor;	
}

nlohmann::json MessageHandler::handle_js_request(JSRequest* request) {
	js::Context js_ctx(&request->screenshot);

	try {
		auto val = js_ctx.eval(request->script);
		if (val.is_bool()) {
			return nlohmann::json({
				{"type", "Boolean"},
				{"value", (bool)val}
			});
		}
		if (val.is_undefined()) {
			return create_error_msg("JS script returned undefined value");
		}
		if (!val.is_object()) {
			return create_error_msg("JS script returned a non-object");
		}

		if (val.is_array()) {
			return create_error_msg("JS script returned an array");
		}

		if (auto tensor = (nn::TextTensor*)val.get_opaque(js::TextTensor::class_id)) {
			return *tensor;
		} else if (auto tensor = (nn::ImgTensor*)val.get_opaque(js::ImgTensor::class_id)) {
			return *tensor;
		} else if (auto point = (nn::Point*)val.get_opaque(js::Point::class_id)) {
			return *point;
		} else {
			return create_error_msg("JS returned a not-supported type");
		}

	} catch (const nn::ContinueError& continue_error) {
		return nlohmann::json({
			{"type", "ContinueError"}
		});
	} catch (const std::exception& err) {
		return create_error_msg(err.what());
	}
}

nlohmann::json MessageHandler::create_error_msg(const std::string& message) {
	nlohmann::json error;
	error["type"] = "Error";
	error["message"] = message;
	return error;
}