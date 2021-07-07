
#include <spdlog/spdlog.h>

#include "MessageHandler.hpp"

#include "../js/Runtime.hpp"
#include "../js/Tensor.hpp"
#include "../js/Point.hpp"

#include "../nn/TextTensor.hpp"
#include "../nn/ImgTensor.hpp"

void MessageHandler::run() {
	while (true) {
		handle_request(channel->receive_request());
	}
}

void MessageHandler::handle_request(std::unique_ptr<Request> request) {
	spdlog::trace(fmt::format("Got the request \n{}", request->to_string()));
	nlohmann::json response;
	if (auto p = dynamic_cast<TextRequest*>(request.get())) {
		response = handle_text_request(p);
	} else if (auto p = dynamic_cast<ImgRequest*>(request.get())) {
		response = handle_img_request(p);
	} else if (auto p = dynamic_cast<JSRequest*>(request.get())) {
		response = handle_js_request(p);
	}

	spdlog::trace(fmt::format("The response is {}", response.dump(4)));
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

	auto val = js_ctx.eval(request->script);
	if (!val.is_object()) {
		//send error
		return {};
	}

	if (val.is_array()) {
		//send error_message
		return {};
	}

	if (val.is_instance_of(js_ctx.get_global_object().get_property_str("TextTensor"))) {
		nn::TextTensor* tensor = (nn::TextTensor*)val.get_opaque(js::TextTensor::class_id);
		return *tensor;
	} else if (val.is_instance_of(js_ctx.get_global_object().get_property_str("ImgTensor"))) {
		nn::ImgTensor* tensor = (nn::ImgTensor*)val.get_opaque(js::ImgTensor::class_id);
		return *tensor;
	} else if (val.is_instance_of(js_ctx.get_global_object().get_property_str("Point"))) {
		nn::Point* point = (nn::Point*)val.get_opaque(js::Point::class_id);
		return *point;
	} else {
		//send error_message
		return {};
	}
}