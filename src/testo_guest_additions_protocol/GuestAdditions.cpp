
#include <coro/Timeout.h>
#include "GuestAdditions.hpp"
#include <os/File.hpp>
#include "base64.hpp"
#include <regex>

using namespace std::literals::chrono_literals;

bool GuestAdditions::is_avaliable(std::chrono::milliseconds time_to_wait) {
	try {
		nlohmann::json request = {
			{"method", "check_avaliable"}
		};

		coro::Timeout timeout(time_to_wait);

		send(std::move(request));

		auto response = recv();
		return response.at("success").get<bool>();
	} catch (const std::exception&) {
		return false;
	}
}

std::string GuestAdditions::get_tmp_dir() {
	nlohmann::json request = {
		{"method", "get_tmp_dir"}
	};

	coro::Timeout timeout(3s);

	send(std::move(request));

	auto response = recv();
	return response.at("result").at("path");
}

void GuestAdditions::copy_to_guest(const fs::path& src, const fs::path& dst) {
	is_avaliable();

	if (fs::is_regular_file(src)) {
		copy_file_to_guest(src, dst);
	} else if (fs::is_directory(src)) {
		copy_dir_to_guest(src, dst);
	} else {
		throw std::runtime_error("Unknown type of file: " + src.generic_string());
	}
}

void GuestAdditions::copy_from_guest(const fs::path& src, const fs::path& dst) {
	nlohmann::json request = {
		{"method", "copy_files_out"}
	};

	request["args"] = nlohmann::json::array();
	request["args"].push_back(src.generic_string());
	request["args"].push_back(dst.generic_string());

	send(std::move(request));

	auto response = recv();

	for (auto& file: response.at("result")) {
		fs::path dst = file.at("path").get<std::string>();
		if (!file.count("content")) {
			fs::create_directories(dst);
			continue;
		}
		fs::create_directories(dst.parent_path());
		os::File f = os::File::open_for_write(dst);
		if (file.at("content").is_string()) {
			auto content_base64 = file.at("content").get<std::string>();
			auto content = base64_decode(content_base64);
			f.write(content.data(), content.size());
		} else {
			uint64_t file_length = 0;
			recv_raw((uint8_t*)&file_length, sizeof(file_length));
			uint64_t i = 0;
			const uint64_t buf_size = 8 * 1024;
			uint8_t buf[buf_size];
			while (i < file_length) {
				uint64_t chunk_size = std::min(buf_size, file_length - i);
				recv_raw(buf, chunk_size);
				f.write(buf, chunk_size);
				i += chunk_size;
			}
		}
	}
}

void GuestAdditions::remove_from_guest(const fs::path& path) {
	// TODO
}

void GuestAdditions::copy_dir_to_guest(const fs::path& src, const fs::path& dst) {
	for (auto& file: fs::directory_iterator(src)) {
		if (fs::is_regular_file(file)) {
			copy_file_to_guest(file, dst / file.path().filename());
		} else if (fs::is_directory(file)) {
			copy_dir_to_guest(file, dst / file.path().filename());
		} else {
			throw std::runtime_error("Unknown type of file: " + fs::path(file).generic_string());
		}
	}
}

int GuestAdditions::execute(const std::string& command,
	const std::function<void(const std::string&)>& callback)
{
	nlohmann::json request = {
			{"method", "execute"},
			{"args", {
				command
			}}
	};

	send(std::move(request));

	while (true) {
		auto response = recv();

		auto result = response.at("result");
		if (result.count("stdout")) {
			std::string output_base64 = result.at("stdout");
			std::vector<uint8_t> output = base64_decode(output_base64);
			if (output.back() != 0) {
				throw std::runtime_error("Expect null-terminated string");
			}
			callback((char*)output.data());
		}
		if (result.at("status").get<std::string>() == "finished") {
			return result.at("exit_code").get<int>();
		}
	}
}

void GuestAdditions::copy_file_to_guest(const fs::path& src, const fs::path& dst) {
	try {
		nlohmann::json request = {
				{"method", "copy_file"},
				{"args", {
					{
						{"path", dst.generic_string()},
					}
				}}
		};

		os::File f = os::File::open_for_read(src);
		if (ver < VersionNumber(2,2,8)) {
			std::vector<uint8_t> fileContents = f.read_all();
			std::string encoded = base64_encode(fileContents.data(), fileContents.size());
			request.at("args")[0]["content"] = encoded;
			send(std::move(request));
		} else {
			request.at("args")[0]["content"] = nullptr;
			send(std::move(request));

			uint64_t file_length = f.size();
			send_raw((uint8_t*)&file_length, sizeof(file_length));
			uint64_t i = 0;
			const uint64_t buf_size = 8 * 1024;
			uint8_t buf[buf_size];
			while (i < file_length) {
				uint64_t chunk_size = std::min(buf_size, file_length - i);
				f.read(buf, chunk_size);
				send_raw(buf, chunk_size);
				i += chunk_size;
			}
		}

		auto response = recv();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Failed to copy host " + src.generic_string() + " to guest " + dst.generic_string()));
	}
}

bool GuestAdditions::mount(const std::string& folder_name, const fs::path& guest_path, bool permanent) {
	nlohmann::json request = {
		{"method", "mount"},
		{"args", {
			{"folder_name", folder_name},
			{"guest_path", guest_path.generic_string()},
			{"permanent", permanent},
		}}
	};
	send(std::move(request));
	auto response = recv();
	return response.at("was_indeed_mounted");
}

nlohmann::json GuestAdditions::get_shared_folder_status(const std::string& folder_name) {
	nlohmann::json request = {
		{"method", "get_shared_folder_status"},
		{"args", {
			{"folder_name", folder_name}
		}}
	};
	send(std::move(request));
	auto response = recv();
	return response.at("result");
}

bool GuestAdditions::umount(const std::string& folder_name, bool permanent) {
	nlohmann::json request = {
		{"method", "umount"},
		{"args", {
			{"folder_name", folder_name},
			{"permanent", permanent},
		}}
	};
	send(std::move(request));
	auto response = recv();
	return response.at("was_indeed_umounted");
}

void GuestAdditions::send(nlohmann::json command) {
	command["version"] = TESTO_VERSION;
	auto command_str = command.dump();
	uint32_t command_length = command_str.length();
	send_raw((uint8_t*)&command_length, sizeof(command_length));
	send_raw((uint8_t*)command_str.data(), command_str.size());
}

nlohmann::json GuestAdditions::recv() {
	uint32_t json_length = 0;
	recv_raw((uint8_t*)&json_length, sizeof(json_length));
	std::string json_str;
	json_str.resize(json_length);
	recv_raw((uint8_t*)json_str.data(), json_str.size());
	nlohmann::json response = nlohmann::json::parse(json_str);

	if (response.count("version")) {
		ver = response.at("version").get<std::string>();
	}

	if (!response.at("success").get<bool>()) {
		if (ver < VersionNumber(2,2,10)) {
			std::string err = response.at("error");
			throw std::runtime_error(err);
		} else {
			std::string err_base64 = response.at("error");
			std::vector<uint8_t> err = base64_decode(err_base64);
			if (err.back() != 0) {
				throw std::runtime_error("Expect null-terminated string");
			}
			throw std::runtime_error((char*)err.data());
		}
	}

	return response;
}

void CLIGuestAdditions::set_var(const std::string& var_name, const std::string& var_value, bool global) {
	nlohmann::json request = {
		{"method", "set_var"},
		{"args", {
			{"var_name", var_name},
			{"var_value", var_value},
			{"global", global},
		}}
	};
	send(std::move(request));
	auto response = recv();
}

std::string CLIGuestAdditions::get_var(const std::string& var_name) {
	nlohmann::json request = {
		{"method", "get_var"},
		{"args", {
			{"var_name", var_name},
		}}
	};
	send(std::move(request));
	auto response = recv();
	return response.at("var_value");
}
