
#include "GuestAdditions.hpp"
#include <coro/Timeout.h>
#include "base64.hpp"
#include <fstream>

using namespace std::literals::chrono_literals;

bool GuestAdditions::is_avaliable() {
	try {
		nlohmann::json request = {
			{"method", "check_avaliable"}
		};

		coro::Timeout timeout(3s);

		send(request);

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

	send(request);

	auto response = recv();
	if(!response.at("success").get<bool>()) {
		throw std::runtime_error(response.at("error").get<std::string>());
	}
	return response.at("result").at("path");
}

void GuestAdditions::copy_to_guest(const fs::path& src, const fs::path& dst) {
	//4) Now we're all set
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

	send(request);

	auto response = recv();

	if(!response.at("success").get<bool>()) {
		throw std::runtime_error(response.at("error").get<std::string>());
	}

	for (auto& file: response.at("result")) {
		fs::path dst = file.at("path").get<std::string>();
		if (!file.count("content")) {
			fs::create_directories(dst);
			continue;
		}
		fs::create_directories(dst.parent_path());
		auto content_base64 = file.at("content").get<std::string>();
		auto content = base64_decode(content_base64);
		std::ofstream file_stream(dst.generic_string(), std::ios::out | std::ios::binary);
		file_stream.write((const char*)&content[0], content.size());
		file_stream.close();
	}
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

	send(request);

	while (true) {
		auto response = recv();
		if (!response.at("success").get<bool>()) {
			throw std::runtime_error(std::string("Negotiator inner error: ") + response.at("error").get<std::string>());
		}

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

		send(request);

		auto response = recv();

		if(!response.at("success").get<bool>()) {
			throw std::runtime_error(response.at("error").get<std::string>());
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Failed to copy host " + src.generic_string() + " to guest " + dst.generic_string()));
	}
}

void GuestAdditions::send(const nlohmann::json& command) {
	auto command_str = command.dump();
	uint32_t command_length = command_str.length();
	size_t n = send_raw((uint8_t*)&command_length, sizeof(command_length));
	if (n != sizeof(command_length)) {
		throw std::runtime_error("Failed to send command_length");
	}
	n = send_raw((uint8_t*)command_str.data(), command_str.size());
	if (n != command_str.size()) {
		throw std::runtime_error("Failed to send command_str");
	}
}

nlohmann::json GuestAdditions::recv() {
	uint32_t json_length = 0;
	size_t n = recv_raw((uint8_t*)&json_length, sizeof(uint32_t));
	if (n != sizeof(uint32_t)) {
		throw std::runtime_error("Failed to read json_length");
	}
	std::string json_str;
	json_str.resize(json_length);
	n = recv_raw((uint8_t*)json_str.data(), json_str.size());
	if (n != json_str.size()) {
		throw std::runtime_error("Failed to read json_str");
	}
	return nlohmann::json::parse(json_str);
}
