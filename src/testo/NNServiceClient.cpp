
#include "NNServiceClient.hpp"

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
			std::this_thread::sleep_for(2s);
		}
		
	}
	
	throw std::runtime_error("Can't connect to the nn_service");
}

nn::TextTensor NNServiceClient::find_text(const stb::Image<stb::RGB>& image,
	std::string text_to_find,
	std::string fg,
	std::string bg)
{
	for (size_t i = 0; i < tries; ++i) {
		try {
			channel->send_request(TextRequest(image, text_to_find, fg, bg));
			auto response = channel->receive_response();
			return response.get<nn::TextTensor>();
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

nn::ImgTensor NNServiceClient::find_img(const stb::Image<stb::RGB>& image,
	const stb::Image<stb::RGB>& ref)
{
	for (size_t i = 0; i < tries; ++i) {
		try {
			channel->send_request(ImgRequest(image, ref));
			auto response = channel->receive_response();
			return response.get<nn::ImgTensor>();
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