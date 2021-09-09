
#pragma once

#include "Channel.hpp"

struct MessageHandler {
	MessageHandler(std::shared_ptr<Channel> channel_): channel(std::move(channel_)) {};
	~MessageHandler() = default;

	MessageHandler(const MessageHandler&) = delete;
	MessageHandler& operator=(const MessageHandler&) = delete;

	void run();

private:
	nlohmann::json handle_request(nlohmann::json& request);

	nlohmann::json handle_js_eval_request(nlohmann::json& request);
	nlohmann::json handle_js_validate_request(nlohmann::json& request);
	std::shared_ptr<Channel> channel;
};
