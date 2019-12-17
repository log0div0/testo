
#include "Server.hpp"
#include "base64.hpp"

#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <cstdlib>
#include <fcntl.h>
#include <sys/stat.h>

Server::Server(const std::string& fd_path): fd_path(fd_path) {}

Server::~Server() {
	if (fd) {
		close(fd);
	}
}

nlohmann::json Server::read() {
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
	fd = open (fd_path.c_str(), O_RDWR);
	if (fd < 0) {
		std::string error_msg = "error " + std::to_string(errno) + " opening " + fd_path + ": " + strerror (errno);
		throw std::runtime_error(error_msg);
	}

	std::cout << "Connected to " << fd_path << std::endl;

	std::cout << "Waiting for commands\n";
	while (true) {
		auto command = read();
		//std::cout << command.dump(4) << std::endl;
		handle_command(command);
	}
}

void Server::handle_command(const nlohmann::json& command) {
	std::string method_name = command.at("method").get<std::string>();

	try {
		if (method_name == "check_avaliable") {
			return handle_check_avaliable();
		} else if (method_name == "copy_file") {
			//std::cout << "Copy file\n";
			return handle_copy_file(command.at("args"));
		} else if (method_name == "copy_files_out") {
			return handle_copy_file_out(command.at("args"));
		} else if (method_name == "execute") {
			return handle_execute(command.at("args"));
		} else {
			throw std::runtime_error(std::string("Method ") + method_name + " is not supported");
		}
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

void Server::handle_check_avaliable() {
	nlohmann::json response = {
		{"success", true},
		{"result", nlohmann::json::object()}
	};

	send(response);
}

std::string get_folder(const std::string& str) {
	size_t found;
	found = str.find_last_of("/");
	return str.substr(0,found);
}

void Server::handle_copy_file(const nlohmann::json& args) {
	for (auto file: args) {
		auto content64 = file.at("content").get<std::string>();
		auto content = base64_decode(content64);
		auto dst = file.at("path").get<std::string>();
		std::cout << "Copying " << dst << std::endl;
		std::string mkdir_cmd = std::string("mkdir -p ") + get_folder(dst);
		std::system(mkdir_cmd.c_str());
		std::ofstream file_stream(dst, std::ios::out | std::ios::binary);
		if (!file_stream) {
			throw std::runtime_error("Couldn't open file stream to write file " + dst);
		}
		file_stream.write((const char*)&content[0], content.size());
		file_stream.close();
		std::cout << "Copied file " << dst << std::endl;
	}

	nlohmann::json response = {
		{"success", true},
		{"result", nlohmann::json::object()}
	};

	send(response);
}

void Server::handle_copy_file_out(const nlohmann::json& args) {
	std::string src = args[0];
	std::string dst = args[1];
	std::cout << "SRC: " << src << std::endl;
	std::cout << "DST: " << dst << std::endl;

	struct stat info;
	if (stat(src.c_str(), &info)) {
		std::string error_msg = "error " + std::to_string(errno) + " stat() " + src + ": " + strerror (errno);
		throw std::runtime_error(error_msg);
	} else if(info.st_mode & S_IFDIR) {
		//copydir
	} else {
		std::ofstream file_stream(dst, std::ios::in | std::ios::binary);
		if (!file_stream) {
			throw std::runtime_error("Couldn't open file stream to read file " + src);
		}
	}

}


void Server::handle_execute(const nlohmann::json& args) {
	auto cmd = args[0].get<std::string>();
	cmd += " 2>&1";

	std::array<char, 256> buffer;
	auto pipe = popen(cmd.c_str(), "r");

	if (!pipe) {
		throw std::runtime_error("popen() failed!");
	}

	while(!feof(pipe)) {
		if (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
			nlohmann::json result = {
				{"success", true},
				{"result", {
					{"status", "pending"},
					{"stdout", buffer.data()}
				}}
			};
			send(result);
		}
	}

	auto rc = pclose(pipe);

	nlohmann::json result = {
		{"success", true},
		{"result", {
			{"status", "finished"},
			{"exit_code", rc}
		}}
	};

	send(result);
}
