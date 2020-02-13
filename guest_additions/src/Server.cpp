
#include "Server.hpp"
#include "Process.hpp"

#include "base64.hpp"

#include <spdlog/spdlog.h>
#include <stdexcept>
#include <fstream>

Server::Server(const fs::path& fd_path_): fd_path(fd_path_) {}

void Server::run() {
	channel = Channel(fd_path);

	spdlog::info("Connected to " + fd_path.generic_string());
	spdlog::info("Waiting for commands");

	while (true) {
		auto command = channel.read();
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

	channel.send(response);
}

void Server::handle_check_avaliable() {
	spdlog::info("Checking avaliability call");

	nlohmann::json response = {
		{"success", true},
		{"result", nlohmann::json::object()}
	};

	channel.send(response);
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

	channel.send(response);
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

	channel.send(result);
	spdlog::info("Copied FROM guest: " + src.generic_string());
}

void Server::handle_execute(const nlohmann::json& args) {
	auto cmd = args[0].get<std::string>();

	spdlog::info("Executing command " + cmd);

	cmd += " 2>&1";

	Process process(cmd);

	while (process.is_running()) {
		std::string output = process.read();
		if (output.size()) {
			nlohmann::json result = {
				{"success", true},
				{"result", {
					{"status", "pending"},
					{"stdout", output.data()}
				}}
			};
			channel.send(result);
		}
	}

	int rc = process.wait();

	nlohmann::json result = {
		{"success", true},
		{"result", {
			{"status", "finished"},
			{"exit_code", rc}
		}}
	};

	channel.send(result);

	spdlog::info("Command finished: " + cmd);
	spdlog::info("Return code: " + std::to_string(rc));
}
