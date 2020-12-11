
#include "GuestAdditions.hpp"
#include <coro/Timeout.h>
#include "base64.hpp"
#include <fstream>
#include <regex>

using namespace std::literals::chrono_literals;


VersionNumber::VersionNumber(const std::string& str) {
	static std::regex regex(R"((\d+).(\d+).(\d+))");
	std::smatch match;
	if (!std::regex_match(str, match, regex)) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
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

bool GuestAdditions::is_avaliable() {
	try {
		nlohmann::json request = {
			{"method", "check_avaliable"}
		};

		coro::Timeout timeout(3s);

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
	if(!response.at("success").get<bool>()) {
		throw std::runtime_error(response.at("error").get<std::string>());
	}
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
		std::ofstream file_stream(dst.generic_string(), std::ios::out | std::ios::binary);
		if (file.at("content").is_string()) {
			auto content_base64 = file.at("content").get<std::string>();
			auto content = base64_decode(content_base64);
			file_stream.write((const char*)&content[0], content.size());
		} else {
			uint64_t file_length = 0;
			recv_raw((uint8_t*)&file_length, sizeof(file_length));
			uint64_t i = 0;
			const uint64_t buf_size = 8 * 1024;
			uint8_t buf[buf_size];
			while (i < file_length) {
				uint64_t chunk_size = std::min(buf_size, file_length - i);
				recv_raw(buf, chunk_size);
				file_stream.write((const char*)buf, chunk_size);
				i += chunk_size;
			}
		}
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

	send(std::move(request));

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
		nlohmann::json request = {
				{"method", "copy_file"},
				{"args", {
					{
						{"path", dst.generic_string()},
					}
				}}
		};

		std::ifstream file(src.generic_string(), std::ios::binary);
		if (ver < VersionNumber(2,2,8)) {
			std::noskipws(file);
			std::vector<uint8_t> fileContents = {std::istream_iterator<uint8_t>(file), std::istream_iterator<uint8_t>()};
			std::string encoded = base64_encode(fileContents.data(), fileContents.size());
			request.at("args")[0]["content"] = encoded;
			send(std::move(request));
		} else {
			request.at("args")[0]["content"] = nullptr;
			send(std::move(request));

			file.seekg(0, std::ios::end);
			uint64_t file_length = file.tellg();
			file.seekg(0, std::ios::beg);

			send_raw((uint8_t*)&file_length, sizeof(file_length));
			uint64_t i = 0;
			const uint64_t buf_size = 8 * 1024;
			uint8_t buf[buf_size];
			while (i < file_length) {
				uint64_t chunk_size = std::min(buf_size, file_length - i);
				file.read((char*)buf, chunk_size);
				send_raw(buf, chunk_size);
				i += chunk_size;
			}
		}

		auto response = recv();

		if(!response.at("success").get<bool>()) {
			throw std::runtime_error(response.at("error").get<std::string>());
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Failed to copy host " + src.generic_string() + " to guest " + dst.generic_string()));
	}
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

	return response;
}
