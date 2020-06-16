
#include "Server.hpp"
#include "process/Process.hpp"
#include "File.hpp"

#include "base64.hpp"

#include <spdlog/spdlog.h>
#include <stdexcept>

Server::Server(const std::string& fd_path_): channel(fd_path_) {
	spdlog::info("Connected to " + fd_path_);
}

void Server::run() {
	spdlog::info("Waiting for commands");

	while (true) {
		try {
			auto command = channel.receive();
			spdlog::info(command.dump(2));
			handle_command(command);
		} catch (const std::exception& error) {
			spdlog::error("Error in Server::run loop");
			spdlog::error(error.what());
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			spdlog::info("Continue handle commands ...");
		}
	}
}

void Server::handle_command(const nlohmann::json& command) {
	std::string method_name = command.at("method").get<std::string>();

	try {
		if (method_name == "check_avaliable") {
			return handle_check_avaliable();
		} else if (method_name == "get_tmp_dir") {
			return handle_get_tmp_dir();
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
		spdlog::error("Error in Server::handle_command method");
		spdlog::error(error.what());
		send_error(error.what());
	}
}

void Server::send_error(const std::string& error) {
	spdlog::info("Sending error " + error);
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

void Server::handle_get_tmp_dir() {
	spdlog::info("Getting tmp dir");

	nlohmann::json response = {
		{"success", true},
		{"result", {
			{"path", fs::temp_directory_path().generic_string()}
		}}
	};

	channel.send(response);
	spdlog::info("Getting tmp dir is OK");
}

void Server::handle_copy_file(const nlohmann::json& args) {
	for (auto file: args) {
		auto content64 = file.at("content").get<std::string>();
		auto content = base64_decode(content64);
		fs::path dst = file.at("path").get<std::string>();
		spdlog::info("Copying file to guest: " + dst.generic_string());

		if (dst.is_relative()) {
			throw std::runtime_error("Destination path on vm must be absolute");
		}

		make_directories(dst.parent_path());
		write_file(dst, content);
		spdlog::info("File copied successfully to guest: " + dst.generic_string());
	}

	nlohmann::json response = {
		{"success", true},
		{"result", nlohmann::json::object()}
	};

	channel.send(response);
}

nlohmann::json Server::copy_single_file_out(const fs::path& src, const fs::path& dst) {
	std::vector<uint8_t> fileContents = read_file(src);
	std::string encoded = base64_encode(fileContents.data(), (uint32_t)fileContents.size());

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

#if __linux__
	cmd += " 2>&1";
#endif

	Process process(cmd);

	while (!process.eof()) {
		std::string output = process.read();
		if (output.size()) {
			nlohmann::json result = {
				{"success", true},
				{"result", {
					{"status", "pending"},
					{"stdout", base64_encode((uint8_t*)output.data(), output.size() + 1)}
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
