
#include "Channel.hpp"
#include <coro/IoService.h>
#include <spdlog/spdlog.h>
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

	spdlog::info("msg_size = {}", msg_size);

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

	spdlog::info("json_str = {}", json_str);

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

void Channel::close() {
	close(fd);
}

#endif

#ifdef WIN32

#include "winapi.hpp"
#include "Channel_getInfo.hpp"

Channel::Channel(const std::string& fd_path):
	stream(asio::windows::stream_handle(coro::IoService::current()->_impl))
{
	HANDLE handle = CreateFile(winapi::utf8_to_utf16(fd_path).c_str(),
		GENERIC_WRITE | GENERIC_READ,
		0,
		NULL,
		OPEN_EXISTING,
		FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
		NULL);
	if (handle == INVALID_HANDLE_VALUE) {
		throw std::runtime_error("CreateFile failed");
	}
	stream.handle().assign(handle);
	info_buf.resize(sizeof(VIRTIO_PORT_INFO));
}

Channel::~Channel() {
}

Channel& Channel::operator=(Channel&& other) {
	std::swap(stream, other.stream);
	std::swap(info_buf, other.info_buf);
	return *this;
}

size_t Channel::read(uint8_t* data, size_t size) {
	PVIRTIO_PORT_INFO info = Channel_getInfo(stream.handle().native_handle(), info_buf);
	if (!info->HostConnected) {
		return 0;
	}
	return stream.readSome(asio::buffer(data, size));
}

size_t Channel::write(uint8_t* data, size_t size) {
	return stream.write(asio::buffer(data, size));
}

void Channel::close() {
	CloseHandle(stream.handle().native_handle());
}

#endif
