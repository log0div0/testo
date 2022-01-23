
#pragma once

#include "testo_nn_server_protocol/Channel.hpp"

struct NNClient {
	NNClient(const std::string& ip, const std::string& port);
	~NNClient();

	NNClient(const NNClient& other) = delete;
	NNClient& operator=(const NNClient& other) = delete;

	nlohmann::json eval_js(const stb::Image<stb::RGB>* image, const std::string& script);

private:
	void establish_connection();
	nlohmann::json receive_response();

	constexpr static int establish_connection_tries = 5;
	constexpr static int rpc_tries = 5;

	void establish_connection_wrapper(const std::function<void()>& fn);
	nlohmann::json rcp_wrapper(const std::function<nlohmann::json()>& fn);

	Endpoint endpoint;
	std::shared_ptr<Channel> channel;
};
