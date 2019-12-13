
#include "Server.hpp"
#include "base64.hpp"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

Server::Server(const std::string& fd_path): fd_path(fd_path) {}

Server::~Server() {
	if (fd) {
		close(fd);
	}
}

void Server::set_interface_attribs(int speed, int parity) {
	struct termios tty;
	if (tcgetattr (fd, &tty) != 0) {
		std::string error_msg = "error " + std::to_string(errno) + " from tcgetattr";
		throw std::runtime_error(error_msg);
	}
	cfsetospeed (&tty, speed);
	cfsetispeed (&tty, speed);
	tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;		// 8-bit chars
	// disable IGNBRK for mismatched speed tests; otherwise receive break
	// as \000 chars
	tty.c_iflag &= ~IGNBRK;		// disable break processing
	tty.c_lflag = 0;			// no signaling chars, no echo,
								// no canonical processing
	tty.c_oflag = 0;		// no remapping, no delays
	tty.c_cc[VMIN]  = 0;	// read doesn't block
	tty.c_cc[VTIME] = 5;	// 0.5 seconds read timeout

	tty.c_iflag &= ~(IXON | IXOFF | IXANY); // shut off xon/xoff ctrl

	tty.c_cflag |= (CLOCAL | CREAD);		// ignore modem controls,
											// enable reading
	tty.c_cflag &= ~(PARENB | PARODD);		// shut off parity
	tty.c_cflag |= parity;
	tty.c_cflag &= ~CSTOPB;
	tty.c_cflag &= ~CRTSCTS;

	if (tcsetattr (fd, TCSANOW, &tty) != 0) {
		std::string error_msg = "error " + std::to_string(errno) + " from tcgetattr";
		throw std::runtime_error(error_msg);
	}
}

void Server::set_blocking(bool should_block) {
	struct termios tty;
	memset (&tty, 0, sizeof tty);
	if (tcgetattr (fd, &tty) != 0) {
		std::string error_msg = "error " + std::to_string(errno) + " from tcgetattr";
		throw std::runtime_error(error_msg);
	}
	tty.c_cc[VMIN]  = should_block ? 1 : 0;
	tty.c_cc[VTIME] = 5;            // 0.5 seconds read timeout
	if (tcsetattr (fd, TCSANOW, &tty) != 0) {
		std::string error_msg = "error " + std::to_string(errno) + " setting term attributes";
		throw std::runtime_error(error_msg);
	}
}

nlohmann::json Server::read() {
	uint32_t msg_size;
	if (::read (fd, &msg_size, 4) != 4) {
		throw std::runtime_error("Can't read msg size");
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

void Server::send(const nlohmann::json& response) {
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

void Server::run() {
	fd = open (fd_path.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
	if (fd < 0) {
		std::string error_msg = "error " + std::to_string(errno) + " opening " + fd_path + ": " + strerror (errno);
		throw std::runtime_error(error_msg);
	}

	std::cout << "Connected to " << fd_path << std::endl;

	set_interface_attribs(B115200, 0);  // set speed to 115,200 bps, 8n1 (no parity)
	set_blocking(true);

	std::cout << "Waiting for commands\n";
	while (true) {
		auto command = read();
		std::cout << command.dump(4) << std::endl;
		handle_command(command);
	}
}

void Server::handle_command(const nlohmann::json& command) {
	std::string method_name = command.at("method").get<std::string>();

	try {
		nlohmann::json result;
		if (method_name == "check_avaliable") {
			result = nlohmann::json::object();
		} else if (method_name == "copy_file") {
			result = handle_copy_file(command.at("args"));
		} else {
			throw std::runtime_error(std::string("Method ") + method_name + " is not supported");
		}

		send_ok(result);
	} catch (const std::exception& error) {
		send_error(error.what());
	}
}

void Server::send_error(const std::string& error) {
	nlohmann::json response = {
		{"success", false},
		{"error", error}
	};

	send(response);
}

void Server::send_ok(const nlohmann::json& result) {
	nlohmann::json response = {
		{"success", true},
		{"result", result}
	};

	send(response);
}

nlohmann::json Server::handle_copy_file(const nlohmann::json& args) {
	for (auto file: args) {
		auto content64 = file.at("content").get<std::string>();
		auto content = base64_decode(content64);
		auto dst = file.at("path").get<std::string>();
		std::ofstream file_stream(dst, std::ios::out | std::ios::binary);
		file_stream.write((const char*)&content[0], content.size());
		file_stream.close();
		std::cout << "Copied file " << dst << std::endl;
	}

	return nlohmann::json::object();
}