
#include "Server.hpp"
#include <os/Process.hpp>
#include <os/File.hpp>

#include "base64.hpp"

#include <spdlog/spdlog.h>
#include <stdexcept>
#include <regex>
#include <fstream>

#ifdef WIN32
#include <winapi/Functions.hpp>
#include <winapi/RegKey.hpp>

std::map<std::string, std::string> get_environment_from_registry() {
	std::map<std::string, std::string> result;
	winapi::RegKey regkey(HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment");
	spdlog::info("ENUM VALUES SIZE = {}", regkey.enum_values().size());
	for (const std::string& name: regkey.enum_values()) {
		result[name] = regkey.get_str(name);
	}
	return result;
}
#endif

VersionNumber::VersionNumber(const std::string& str) {
	static std::regex regex(R"((\d+).(\d+).(\d+))");
	std::smatch match;
	if (!std::regex_match(str, match, regex)) {
		throw std::runtime_error("Failed to parse VersionNumber");
	}
	MAJOR = stoi(match[1]);
	MINOR = stoi(match[2]);
	PATCH = stoi(match[3]);
}

bool VersionNumber::operator<(const VersionNumber& other) {
	if (MAJOR < other.MAJOR) {
		return true;
	}else if (MAJOR == other.MAJOR) {
		if (MINOR < other.MINOR) {
			return true;
		} else if (MINOR == other.MINOR) {
			return PATCH < other.PATCH;
		} else {
			return false;
		}
	} else {
		return false;
	}
}

std::string VersionNumber::to_string() const {
	return std::to_string(MAJOR) + "." + std::to_string(MINOR) + "." + std::to_string(PATCH);
}

Server::Server(const std::string& fd_path_): channel(fd_path_) {
	spdlog::info("Connected to " + fd_path_);
}

void Server::run() {
	spdlog::info("Waiting for commands");

	while (!is_canceled) {
		try {
			auto command = channel.receive();
			// spdlog::info(command.dump(2));
			handle_command(command);
		} catch (const std::exception& error) {
			spdlog::error("Error in Server::run loop");
			spdlog::error(error.what());
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			spdlog::info("Continue handle commands ...");
		}
	}
}

void Server::force_cancel() {
	spdlog::info("Force cancel");
	is_canceled = true;
	channel.close();
}

void Server::handle_command(const nlohmann::json& command) {
	std::string method_name = command.at("method").get<std::string>();

	if (command.count("version")) {
		ver = command.at("version").get<std::string>();
	} else {
		ver = VersionNumber();
	}

	spdlog::info("testo version = {}, testo guest additions version = {}", ver.to_string(), TESTO_VERSION);

	try {
		if (method_name == "check_avaliable") {
			return handle_check_avaliable(command);
		} else if (method_name == "get_tmp_dir") {
			return handle_get_tmp_dir(command);
		} else if (method_name == "copy_file") {
			return handle_copy_file(command);
		} else if (method_name == "copy_files_out") {
			return handle_copy_files_out(command);
		} else if (method_name == "execute") {
			return handle_execute(command);
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
	nlohmann::json response = {
		{"success", false},
	};

	if (ver < VersionNumber(2,2,10)) {
		response["error"] = error;
	} else {
		response["error"] = base64_encode((uint8_t*)error.data(), error.size() + 1);
	}

	channel.send(std::move(response));
}

void Server::handle_check_avaliable(const nlohmann::json& command) {
	spdlog::info("Checking avaliability call");

	nlohmann::json response = {
		{"success", true},
		{"result", nlohmann::json::object()}
	};

	channel.send(std::move(response));
	spdlog::info("Checking avaliability is OK");
}

void Server::handle_get_tmp_dir(const nlohmann::json& command) {
	spdlog::info("Getting tmp dir");

	nlohmann::json response = {
		{"success", true},
		{"result", {
			{"path", fs::temp_directory_path().generic_string()}
		}}
	};

	channel.send(std::move(response));
	spdlog::info("Getting tmp dir is OK");
}

void Server::handle_copy_file(const nlohmann::json& command) {
	const nlohmann::json& args = command.at("args");

	for (auto file: args) {
		fs::path dst = file.at("path").get<std::string>();
		spdlog::info("Copying file to guest: " + dst.generic_string());

		if (dst.is_relative()) {
			throw std::runtime_error("Destination path on vm must be absolute");
		}

		if (!fs::exists(dst.parent_path())) {
			if (!fs::create_directories(dst.parent_path())) {
				throw std::runtime_error("Can't create directory: " + dst.parent_path().generic_string());
			}
		}

		os::File f = os::File::open_for_write(dst.generic_string());
		if (file.at("content").is_string()) {
			auto content64 = file.at("content").get<std::string>();
			auto content = base64_decode(content64);
			f.write(content.data(), content.size());
		} else {
			uint64_t file_length = 0;
			channel.receive_raw((uint8_t*)&file_length, sizeof(file_length));
			uint64_t i = 0;
			const uint64_t buf_size = 8 * 1024;
			uint8_t buf[buf_size];
			while (i < file_length) {
				uint64_t chunk_size = std::min(buf_size, file_length - i);
				channel.receive_raw(buf, chunk_size);
				f.write(buf, chunk_size);
				i += chunk_size;
			}
		}

		spdlog::info("File copied successfully to guest: " + dst.generic_string());
	}

	nlohmann::json response = {
		{"success", true},
		{"result", nlohmann::json::object()}
	};

	channel.send(std::move(response));
}

nlohmann::json Server::copy_single_file_out(const fs::path& src, const fs::path& dst) {
	if (src.is_relative()) {
		throw std::runtime_error(fmt::format("Source path on vm must be absolute"));
	}

	nlohmann::json result = {
		{"src", src.generic_string()},
		{"path", dst.generic_string()},
	};

	if (ver < VersionNumber(2,2,8)) {
		os::File file = os::File::open_for_read(src.generic_string());
		std::vector<uint8_t> fileContents = file.read_all();
		std::string encoded = base64_encode(fileContents.data(), (uint32_t)fileContents.size());
		result["content"] = std::move(encoded);
	} else {
		result["content"] = nullptr;
	}

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

void Server::handle_copy_files_out(const nlohmann::json& command) {
	const nlohmann::json& args = command.at("args");

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

	channel.send(std::move(result));

	if (ver < VersionNumber(2,2,8)) {
		// do nothing
	} else {
		for (auto& item: files) {
			if (!item.count("content")) {
				continue;
			}
			std::string path = item.at("src");
			os::File file = os::File::open_for_read(path);
			uint64_t file_length = file.size();
			spdlog::info("Sending file {}, file size = {}", path, file_length);
			channel.send_raw((uint8_t*)&file_length, sizeof(file_length));
			uint64_t i = 0;
			const uint64_t buf_size = 8 * 1024;
			uint8_t buf[buf_size];
			while (i < file_length) {
				uint64_t chunk_size = std::min(buf_size, file_length - i);
				file.read(buf, chunk_size);
				channel.send_raw(buf, chunk_size);
				i += chunk_size;
			}
		}
	}

	spdlog::info("Copied FROM guest: " + src.generic_string());
}

void Server::handle_execute(const nlohmann::json& command) {
	const nlohmann::json& args = command.at("args");

	auto cmd = args[0].get<std::string>();

	spdlog::info("Executing command " + cmd);

#if __linux__
	cmd += " 2>&1";
#endif

#if WIN32
	std::map<std::string, std::string> env_vars = winapi::get_environment_strings();
	for (auto& kv: get_environment_from_registry()) {
		env_vars[kv.first] = kv.second;
	}
	os::Process process(cmd, &env_vars);
#else
	os::Process process(cmd);
#endif

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
			channel.send(std::move(result));
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

	channel.send(std::move(result));

	spdlog::info("Command finished: " + cmd);
	spdlog::info("Return code: " + std::to_string(rc));
}
