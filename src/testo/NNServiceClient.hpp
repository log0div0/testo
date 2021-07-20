
#pragma once

#include "../nn_service/Channel.hpp"
#include "../nn/TextTensor.hpp"
#include "../nn/ImgTensor.hpp"

struct NNServiceClient {
	NNServiceClient() = default;

	NNServiceClient(const std::string& ip, const std::string& port):
		endpoint(asio::ip::address::from_string(ip), std::stoul(port)),
		channel(new Channel(Socket()))
	{		
		establish_connection();
	}

	nn::TextTensor find_text(const stb::Image<stb::RGB>* image,
		std::string text_to_find = "",
		std::string fg = "",
		std::string bg = "");

	nn::ImgTensor find_img(const stb::Image<stb::RGB>* image, const stb::Image<stb::RGB>* ref);


private:
	constexpr static int tries = 5;
	void establish_connection();

	Endpoint endpoint;
	std::shared_ptr<Channel> channel;
};
