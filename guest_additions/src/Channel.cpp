
#include "Channel.hpp"
#ifdef WIN32
#include "winapi.hpp"
#endif

#include <thread>

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

	std::string json_str;
	json_str.resize(msg_size);

	size_t already_read = 0;

	while (already_read < msg_size) {
		size_t n = read((uint8_t*)&json_str[already_read], msg_size - already_read);
		if (n == 0) {
			throw std::runtime_error("EOF while reading");
		}
		already_read += n;
	}

	nlohmann::json result = nlohmann::json::parse(json_str);
	return result;
}

void Channel::send(const nlohmann::json& response) {
	auto response_str = response.dump();
	std::vector<uint8_t> buffer;
	uint32_t response_size = (uint32_t)response_str.length();
	buffer.reserve(sizeof(uint32_t) + response_str.length());
	std::copy((uint8_t*)&response_size, (uint8_t*)(&response_size) + sizeof(uint32_t), std::back_inserter(buffer));

	std::copy(response_str.begin(), response_str.end(), std::back_inserter(buffer));

	size_t already_send = 0;

	while (already_send < buffer.size()) {
		size_t n = write(&buffer[already_send], buffer.size() - already_send);
		if (n == 0) {
			throw std::runtime_error("EOF while writing");
		}
		already_send += n;
	}
}

#ifdef __linux__

#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

Channel::Channel(const std::string& fd_path) {
	fd = open (fd_path.c_str(), O_RDWR);
	if (fd < 0) {
		std::string error_msg = "error " + std::to_string(errno) + " opening " + fd_path + ": " + strerror (errno);
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

size_t Channel::read(uint8_t* data, size_t size) {
	int n = ::read(fd, data, size);
	if (n < 0) {
		throw std::runtime_error("::read failed");
	}
	return n;
}

size_t Channel::write(uint8_t* data, size_t size) {
	int n = ::write(fd, data, size);
	if (n < 0) {
		throw std::runtime_error("::write failed");
	}
	return n;
}

#endif

#ifdef WIN32

Channel::Channel(const std::string& fd_path) {
	handle = CreateFile(winapi::utf8_to_utf16(fd_path).c_str(),
		GENERIC_WRITE | GENERIC_READ,
		0,
		NULL,
		OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL,
		NULL);

	if (handle == INVALID_HANDLE_VALUE) {
		throw std::runtime_error("CreateFileA failed");
	}
}

Channel::~Channel() {
	if (handle) {
		CloseHandle(handle);
		handle = NULL;
	}
}

Channel& Channel::operator=(Channel&& other) {
	std::swap(handle, other.handle);
	return *this;
}

size_t Channel::read(uint8_t* data, size_t size) {
	DWORD result = 0;
	bool success = ReadFile(handle, data, (DWORD)size, &result, NULL);
	if (!success) {
		throw std::runtime_error("ReadFile failed");
	}
	return result;
}

size_t Channel::write(uint8_t* data, size_t size) {
	DWORD result = 0;
	bool success = WriteFile(handle, data, (DWORD)size, &result, NULL);
	if (!success) {
		throw std::runtime_error("WriteFile failed");
	}
	return result;
}

#endif
