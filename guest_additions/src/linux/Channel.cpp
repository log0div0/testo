
#include "../Channel.hpp"

#include <thread>

#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

Channel::Channel(const fs::path& fd_path) {
	fd = open (fd_path.c_str(), O_RDWR);
	if (fd < 0) {
		std::string error_msg = "error " + std::to_string(errno) + " opening " + fd_path.generic_string() + ": " + strerror (errno);
		throw std::runtime_error(error_msg);
	}
}

Channel::~Channel() {
	if (fd >= 0) {
		close(fd);
		fd = -1;
	}
}

Channel& Channel::operator=(Channel&& other) {
	std::swap(fd, other.fd);
	return *this;
}

nlohmann::json Channel::read() {
	uint32_t msg_size;
	while (true) {
		int bytes_read = ::read (fd, &msg_size, 4);
		if (bytes_read == 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		} else if (bytes_read < 0) {
			throw std::runtime_error(std::string("Got error from reading socket: ") + strerror(errno));
		} else if (bytes_read != 4) {
			throw std::runtime_error("Can't read msg size");
		} else {
			break;
		}
	}

	std::string json_str;
	json_str.resize(msg_size);

	int already_read = 0;

	while (already_read < msg_size) {
		int n = ::read (fd, &json_str[already_read], msg_size - already_read);
		if (n == 0) {
			throw std::runtime_error("EOF while reading");
		} else if (n < 0) {
			throw std::runtime_error(std::string("Got error from reading socket: ") + strerror(errno));
		}
		already_read += n;
	}

	nlohmann::json result = nlohmann::json::parse(json_str);
	return result;
}

void Channel::send(const nlohmann::json& response) {
	auto response_str = response.dump();
	std::vector<uint8_t> buffer;
	uint32_t response_size = response_str.length();
	buffer.reserve(sizeof(uint32_t) + response_str.length());
	std::copy((uint8_t*)&response_size, (uint8_t*)(&response_size) + sizeof(uint32_t), std::back_inserter(buffer));

	std::copy(response_str.begin(), response_str.end(), std::back_inserter(buffer));

	int already_send = 0;

	while (already_send < buffer.size()) {
		int n = ::write (fd, &buffer[already_send], buffer.size() - already_send);
		if (n == 0) {
			throw std::runtime_error("EOF while writing");
		} else if (n < 0) {
			throw std::runtime_error(std::string("Got error while sending to socket: ") + strerror(errno));
		}
		already_send += n;
	}
}
