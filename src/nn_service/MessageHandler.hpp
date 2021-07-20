
#pragma once

#include "Channel.hpp"

struct MessageHandler {
	MessageHandler(std::shared_ptr<Channel> channel_): channel(std::move(channel_)) {};
	~MessageHandler() = default;

	MessageHandler(const MessageHandler&) = delete;
	MessageHandler& operator=(const MessageHandler&) = delete;

	void run();

private:
	void handle_request(std::unique_ptr<Message> request);

	nlohmann::json handle_js_request(JSRequest* request);
	std::shared_ptr<Channel> channel;

	nlohmann::json create_error_msg(const std::string& message);
};
