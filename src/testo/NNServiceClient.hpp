
#pragma once

#include "../nn_service/Channel.hpp"

struct NNServiceClient {
	NNServiceClient() = default;

	NNServiceClient(const std::string& ip, const std::string& port):
		endpoint(asio::ip::address::from_string(ip), std::stoul(port)),
		channel(new Channel(Socket()))
	{		
		establish_connection();
	}

	nlohmann::json eval_js(const stb::Image<stb::RGB>* image, const std::string& script);
	nlohmann::json validate_js(const std::string& script);


private:
	constexpr static int tries = 5;
	void establish_connection();

	Endpoint endpoint;
	std::shared_ptr<Channel> channel;
};
