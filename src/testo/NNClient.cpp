
#include "NNClient.hpp"
#include "Logger.hpp"
#include "Exceptions.hpp"

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

bool is_connection_lost(const std::error_code& code) {
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
	return establish_connection_wrapper([&] {
		channel->socket = Socket();
		channel->socket.connect(endpoint);
	});
}

nlohmann::json NNClient::receive_response() {
	nlohmann::json response = channel->recv();
	std::string type = response.at("type");
	if (type == ERROR_RESPONSE) {
		std::string message = response.at("data").get<std::string>();
		throw std::runtime_error(message);
	} else if (type == CONTINUE_ERROR_RESPONSE) {
		std::string message = response.at("data").get<std::string>();
		throw ContinueError(message);
	}
	return response;
}

void NNClient::establish_connection_wrapper(const std::function<void()>& fn) {
	for (size_t i = 0; i < establish_connection_tries; ++i) {
		try {
			return fn();
		} catch (const std::exception& error) {
			std::cerr << error.what() << std::endl;
			if (i < (establish_connection_tries - 1)) {
				std::cerr << "Failed to connect to the server, reconnecting ...\n";
				coro::Timer timer;
				timer.waitFor(2s);
			}
		}
	}

	throw std::runtime_error("Exceeding the number of attempts to connect to the server");
}

nlohmann::json NNClient::rcp_wrapper(const std::function<nlohmann::json()>& fn) {
	for (size_t i = 0; i < rpc_tries; ++i) {
		try {
			return fn();
		} catch (const std::system_error& error) {
			if (is_connection_lost(error.code())) {
				std::cerr << error.what() << std::endl;
				if (i < (rpc_tries - 1)) {
					std::cerr << "Lost the connection to the server, reconnecting...\n";
					establish_connection();
				}
			} else {
				throw;
			}
		}
	}
	throw std::runtime_error("Exceeding the number of attempts to execute RPC");
}

nlohmann::json NNClient::eval_js(const stb::Image<stb::RGB>* image, const std::string& script) {
	return rcp_wrapper([&] {
		channel->send(create_js_eval_request(*image, script));

		while (true) {
			nlohmann::json response = receive_response();
			std::string type = response.at("type");
			if (type == REF_IMAGE_REQUEST) {
				std::string ref_file_path = response.at("data");

				stb::Image<stb::RGB> ref_image;
				try {
					ref_image = stb::Image<stb::RGB>(ref_file_path);
				} catch (const std::exception& error) {
					std::throw_with_nested(std::runtime_error("NN server requested image " + ref_file_path + " but we failed to open the file"));
				}

				channel->send(create_ref_image_response(ref_image));
				continue;
			} else if (type == JS_EVAL_RESPONSE) {
				return response;
			} else {
				throw std::runtime_error(std::string("Unknown message type: ") + type);
			}
		}
	});
}
