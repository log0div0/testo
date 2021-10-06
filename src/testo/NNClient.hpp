
#pragma once

#include "testo_nn_server_protocol/Channel.hpp"

struct NNClient {
	NNClient(const std::string& ip, const std::string& port);
	~NNClient();

	NNClient(const NNClient& other) = delete;
	NNClient& operator=(const NNClient& other) = delete;

	nlohmann::json eval_js(const stb::Image<stb::RGB>* image, const std::string& script);
	nlohmann::json validate_js(const std::string& script);

private:
	constexpr static int tries = 5;
	void establish_connection();

	Endpoint endpoint;
	std::shared_ptr<Channel> channel;
};
