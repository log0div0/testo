
#include "Channel.hpp"
#include <thread>
#include <chrono>

nlohmann::json Channel::receive() {
	uint32_t msg_size;
	while (true) {
		size_t bytes_read = read((uint8_t*)&msg_size, 4);
		if (bytes_read == 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		} else if (bytes_read != 4) {
			throw std::runtime_error("Can't read msg size");
		} else {
			break;
		}
	}

	// spdlog::info("msg_size = {}", msg_size);

	std::string json_str;
	json_str.resize(msg_size);
	receive_raw((uint8_t*)json_str.data(), json_str.size());

	// spdlog::info("json_str = {}", json_str);

	nlohmann::json result = nlohmann::json::parse(json_str);
	return result;
}

void Channel::send(nlohmann::json response) {
	response["version"] = TESTO_VERSION;
	auto response_str = response.dump();
	uint32_t response_size = (uint32_t)response_str.size();
	send_raw((uint8_t*)&response_size, sizeof(response_size));
	send_raw((uint8_t*)response_str.data(), response_size);
}

void Channel::receive_raw(uint8_t* data, size_t size) {
	size_t already_read = 0;
	while (already_read < size) {
		size_t n = read(&data[already_read], size - already_read);
		if (n == 0) {
			throw std::runtime_error("EOF while reading");
		}
		already_read += n;
	}
}

void Channel::send_raw(uint8_t* data, size_t size) {
	size_t already_send = 0;
	while (already_send < size) {
		size_t n = write(&data[already_send], size - already_send);
		if (n == 0) {
			throw std::runtime_error("EOF while writing");
		}
		already_send += n;
	}
}
