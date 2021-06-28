
#pragma once

#include "Channel.hpp"

struct MessageHandler {
	MessageHandler(std::shared_ptr<Channel> channel_): channel(std::move(channel_)) {};
	~MessageHandler() = default;

	MessageHandler(const MessageHandler&) = delete;
	MessageHandler& operator=(const MessageHandler&) = delete;

	void run();

private:
	void handle_request(std::unique_ptr<Request> request);

	nlohmann::json handle_text_request(TextRequest* request);
	nlohmann::json handle_img_request(ImgRequest* request);

	std::shared_ptr<Channel> channel;
};
