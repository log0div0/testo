
#include "Server.hpp"
#include "base64.hpp"
#include <spdlog/spdlog.h>

#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

Server::Server(const fs::path& fd_path): fd_path(fd_path) {}

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
		std::string error_msg = "error " + std::to_string(errno) + " opening " + fd_path.generic_string() + ": " + strerror (errno);
		throw std::runtime_error(error_msg);
	}

	spdlog::info("Connected to " + fd_path.generic_string());
	spdlog::info("Waiting for commands");

	while (true) {
		auto command = read();
		handle_command(command);
	}
}

void Server::handle_command(const nlohmann::json& command) {
	std::string method_name = command.at("method").get<std::string>();

	try {
		if (method_name == "check_avaliable") {
			return handle_check_avaliable();
		} else if (method_name == "copy_file") {
			return handle_copy_file(command.at("args"));
		} else if (method_name == "copy_files_out") {
			return handle_copy_files_out(command.at("args"));
		} else if (method_name == "execute") {
			return handle_execute(command.at("args"));
		} else {
			throw std::runtime_error(std::string("Method ") + method_name + " is not supported");
		}
	} catch (const std::exception& error) {
		spdlog::error(error.what());
		send_error(error.what());
	}
}

void Server::send_error(const std::string& error) {
	spdlog::error("Sending error " + error);
	nlohmann::json response = {
		{"success", false},
		{"error", error}
	};

	send(response);
}

void Server::handle_check_avaliable() {
	spdlog::info("Checking avaliability call");

	nlohmann::json response = {
		{"success", true},
		{"result", nlohmann::json::object()}
	};

	send(response);
	spdlog::info("Checking avaliability is OK");
}

void Server::handle_copy_file(const nlohmann::json& args) {
	for (auto file: args) {
		auto content64 = file.at("content").get<std::string>();
		auto content = base64_decode(content64);
		fs::path dst = file.at("path").get<std::string>();
		spdlog::info("Copying file to guest: " + dst.generic_string());

		if (!fs::exists(dst.parent_path())) {
			if (!fs::create_directories(dst.parent_path())) {
				throw std::runtime_error(std::string("Can't create directory: ") + dst.parent_path().generic_string());
			}
		}

		std::ofstream file_stream(dst, std::ios::out | std::ios::binary);
		if (!file_stream) {
			throw std::runtime_error("Couldn't open file stream to write file " + dst.generic_string());
		}
		file_stream.write((const char*)&content[0], content.size());
		file_stream.close();
		spdlog::info("File copied successfully to guest: " + dst.generic_string());
	}

	nlohmann::json response = {
		{"success", true},
		{"result", nlohmann::json::object()}
	};

	send(response);
}

nlohmann::json Server::copy_single_file_out(const fs::path& src, const fs::path& dst) {
	std::ifstream testFile(src.generic_string(), std::ios::binary);

	std::noskipws(testFile);
	std::vector<uint8_t> fileContents = {std::istream_iterator<uint8_t>(testFile), std::istream_iterator<uint8_t>()};
	std::string encoded = base64_encode(fileContents.data(), fileContents.size());

	nlohmann::json request = {
			{"method", "copy_file"},
			{"args", {
				{
					{"path", dst.generic_string()},
					{"content", encoded}
				}
			}}
	};

	nlohmann::json result = {
		{"path", dst.generic_string()},
		{"content", encoded}
	};

	return result;
}

nlohmann::json Server::copy_directory_out(const fs::path& dir, const fs::path& dst) {
	nlohmann::json files = nlohmann::json::array();

	files.push_back({
		{"path", dst.generic_string()}
	});

	for (auto& file: fs::directory_iterator(dir)) {
		if (fs::is_regular_file(file)) {
			files.push_back(copy_single_file_out(file, dst / fs::path(file).filename()));
		} else if (fs::is_directory(file)) {
			auto result = copy_directory_out(file, dst / fs::path(file).filename());
			files.insert(files.end(), result.begin(), result.end());
		} else {
			throw std::runtime_error("Unknown type of file: " + fs::path(file).generic_string());
		}
	}

	return files;
}

void Server::handle_copy_files_out(const nlohmann::json& args) {
	nlohmann::json files = nlohmann::json::array();
	fs::path src = args[0].get<std::string>();
	fs::path dst = args[1].get<std::string>();

	spdlog::info("Copying FROM guest: " + src.generic_string());

	if (!fs::exists(src)) {
		throw std::runtime_error("Source " + src.generic_string() + " doesn't exist on guest");
	}

	if (fs::is_regular_file(src)) {
		files.push_back(copy_single_file_out(src, dst));
	} else if (fs::is_directory(src)) {
		auto result = copy_directory_out(src, dst);
		files.insert(files.end(), result.begin(), result.end());
	} else {
		throw std::runtime_error("Unknown type of file: " + src.generic_string());
	}

	nlohmann::json result = {
		{"success", true},
		{"result", files}
	};

	send(result);
	spdlog::info("Copied FROM guest: " + src.generic_string());
}


void Server::handle_execute(const nlohmann::json& args) {
	auto cmd = args[0].get<std::string>();

	spdlog::info("Executing command " + cmd);

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

	spdlog::info("Command finished: " + cmd);
	spdlog::info("Return code: " + std::to_string(rc));
}
