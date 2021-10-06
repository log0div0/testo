
#include "NNClient.hpp"
#include "Logger.hpp"

#include <coro/Timer.h>
#include <iostream>

using namespace std::chrono_literals;

NNClient::NNClient(const std::string& ip, const std::string& port):
	endpoint(asio::ip::address::from_string(ip), std::stoul(port)),
	channel(new Channel(Socket()))
{
	TRACE();

	establish_connection();
}

NNClient::~NNClient() {
	TRACE();
}

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

void NNClient::establish_connection() {
	for (size_t i = 1; i <= tries; ++i) {
		try {
			channel->socket = Socket();
			channel->socket.connect(endpoint);
			return;
		} catch (const std::exception& error) {
			std::cerr << error.what() << std::endl;
			coro::Timer timer;
			timer.waitFor(2s);
		}
		
	}
	
	throw std::runtime_error("Can't connect to the nn_server");
}

nlohmann::json NNClient::eval_js(const stb::Image<stb::RGB>* image, const std::string& script)
{
	for (size_t i = 0; i < tries; ++i) {
		try {
			channel->send(create_js_eval_request(*image, script));

			while (true) {
				auto response = channel->recv();
				auto type = response.at("type").get<std::string>();
				if (type == "error") {
					throw std::runtime_error(response.at("data").get<std::string>());
				}
				if (type == "ref_image_request") {
					//handle this
					std::string ref_file_path = response.at("data").get<std::string>();

					stb::Image<stb::RGB> ref_image;
					try {
						ref_image = stb::Image<stb::RGB>(ref_file_path);
					} catch (const std::exception& error) {
						try {
							channel->send(create_error_message(error.what()));
						} catch (...) {}
						throw;
					}

					channel->send(create_ref_image_message(ref_image));
					continue;
				}

				return response;
			}

			return {};
		} catch (const std::system_error& error) {
			if (check_system_code(error.code())) {
				std::cerr << "Lost the connection to the nn_server, reconnecting...\n";
				establish_connection();
			} else {
				throw;
			}
		}
	}
	throw std::runtime_error("Can't request the nn_server to find the text");
}

nlohmann::json NNClient::validate_js(const std::string& script)
{
	for (size_t i = 0; i < tries; ++i) {
		try {
			channel->send(create_js_validate_request(script));
			auto response = channel->recv();
			return response;
		} catch (const std::system_error& error) {
			if (check_system_code(error.code())) {
				std::cerr << "Lost the connection to the nn_server, reconnecting...\n";
				establish_connection();
			} else {
				throw;
			}
		}
	}
	throw std::runtime_error("Can't request the nn_server to find the text");
}
