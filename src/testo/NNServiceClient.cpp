
#include "NNServiceClient.hpp"

#include "coro/Timer.h"

#include <iostream>

using namespace std::chrono_literals;

bool check_system_code(const std::error_code& code) {
	int value = code.value();
	if (value == ECONNABORTED ||
		value == ECONNRESET ||
		value == ENETDOWN ||
		value == ENETRESET ||
		value == ENOENT ||
		value == EPIPE ||
		value == ENOTCONN)
	{
		return true;
	} else {
		return false;
	}
}

void NNServiceClient::establish_connection() {
	for (size_t i = 1; i <= tries; ++i) {
		try {
			std::cout << "Attempt #" << i << " connecting to the nn_service..." << std::endl;
			channel->socket = Socket();
			channel->socket.connect(endpoint);
			std::cout << "Connected!\n";
			return;
		} catch (const std::exception& error) {
			std::cout << error.what() << std::endl;
			coro::Timer timer;
			timer.waitFor(2s);
		}
		
	}
	
	throw std::runtime_error("Can't connect to the nn_service");
}

nlohmann::json NNServiceClient::eval_js(const stb::Image<stb::RGB>* image, const std::string& script)
{
	for (size_t i = 0; i < tries; ++i) {
		try {
			channel->send_request(JSRequest(*image, script));

			while (true) {
				auto response = channel->receive_response();
				auto type = response.at("type").get<std::string>();
				if (type == "ref_image_request") {
					//handle this
					std::string ref_file_path = response.at("data").get<std::string>();
					stb::Image<stb::RGB> ref_image(ref_file_path);
					channel->send_request(RefImage(ref_image));
					continue;
				}

				return response;
			}

			return {};
		} catch (const std::system_error& error) {
			if (check_system_code(error.code())) {
				std::cout << "Lost the connection to the nn_service, reconnecting...\n";
				establish_connection();
			} else {
				throw;
			}
		}
	}
	throw std::runtime_error("Can't request the nn_service to find the text");
}
