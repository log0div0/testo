
#pragma once

#include "testo_nn_server_protocol/Channel.hpp"
#include "js/Context.hpp"

struct MessageHandler: js::ContextEnv {
	MessageHandler(std::shared_ptr<Channel> channel_): channel(std::move(channel_)) {};
	~MessageHandler() = default;

	MessageHandler(const MessageHandler&) = delete;
	MessageHandler& operator=(const MessageHandler&) = delete;

	void run();

	stb::Image<stb::RGB> get_ref_image(const std::string& img_path) override;

private:
	nlohmann::json handle_request(nlohmann::json& request);

	nlohmann::json handle_js_eval_request(nlohmann::json& request);
	nlohmann::json handle_js_validate_request(nlohmann::json& request);
	std::shared_ptr<Channel> channel;
};
