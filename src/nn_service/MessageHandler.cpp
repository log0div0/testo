
#include "MessageHandler.hpp"

#include "../nn/TextTensor.hpp"

void MessageHandler::run() {
	while (true) {
		handle_request(channel->receive_request());
	}
}

void MessageHandler::handle_request(std::unique_ptr<Request> request) {
	if (auto p = dynamic_cast<TextRequest*>(request.get())) {
		handle_text_request(p);
	} else if (auto p = dynamic_cast<ImgRequest*>(request.get())) {
		handle_img_request(p);
	}
}

void MessageHandler::handle_text_request(TextRequest* request) {
	std::cout << "Received text request\n";
	nn::TextTensor tensor = nn::find_text(&request->screenshot);
	if (request->has_text()) {
		tensor = tensor.match_text(&request->screenshot, request->text());
	}

	if (request->has_fg() || request->has_bg()) {
		tensor = tensor.match_color(&request->screenshot, request->color_fg(), request->color_bg());
	}
}

void MessageHandler::handle_img_request(ImgRequest* request) {
	std::cout << "Received img request\n";
}