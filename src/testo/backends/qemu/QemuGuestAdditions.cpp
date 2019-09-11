
#include "QemuGuestAdditions.hpp"
#include "base64.hpp"
#include <fmt/format.h>
#include <fstream>

using namespace std::literals::chrono_literals;

QemuGuestAdditions::QemuGuestAdditions(vir::Domain& domain) {
	auto config = domain.dump_xml();

	auto devices = config.first_child().child("devices");

	std::string path;

	for (auto channel = devices.child("channel"); channel; channel = channel.next_sibling("channel")) {
		if (std::string(channel.child("target").attribute("name").value()) == "negotiator.0") {
			path = std::string(channel.child("source").attribute("path").value());
			break;
		}
	}

	if (!path.length()) {
		throw std::runtime_error("Can't find negotiator channel unix file");
	}

	endpoint = Endpoint(path);

	socket.connect(endpoint);
}

bool QemuGuestAdditions::is_avaliable() {
	try {
		nlohmann::json request = {
			{"method", "check_avaliable"}
		};

		coro::Timeout timeout(3s);

		send(request);

		auto response = recv();
		return response.at("success").get<bool>();

	} catch (...) {
		return false;
	}
}

void QemuGuestAdditions::copy_to_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_seconds) {
	auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(timeout_seconds);
	//4) Now we're all set
	if (fs::is_regular_file(src)) {
		copy_file_to_guest(src, dst, deadline);
	} else if (fs::is_directory(src)) {
		copy_dir_to_guest(src, dst, deadline);
	} else {
		throw std::runtime_error("Unknown type of file: " + src.generic_string());
	}
}

void QemuGuestAdditions::copy_from_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_seconds) {
	nlohmann::json request = {
		{"method", "copy_files_out"}
	};

	request["args"] = nlohmann::json::array();
	request["args"].push_back(src.generic_string());
	request["args"].push_back(dst.generic_string());

	auto chrono_seconds = std::chrono::seconds(timeout_seconds);
	coro::Timeout timeout(chrono_seconds); //actually, it really depends on file size, TODO

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

void QemuGuestAdditions::copy_dir_to_guest(const fs::path& src, const fs::path& dst, std::chrono::system_clock::time_point deadline) {
	for (auto& file: fs::directory_iterator(src)) {
		if (fs::is_regular_file(file)) {
			copy_file_to_guest(file, dst / file.path().filename(), deadline);
		} else if (fs::is_directory(file)) {
			copy_dir_to_guest(file, dst / file.path().filename(), deadline);
		} else {
			throw std::runtime_error("Unknown type of file: " + fs::path(file).generic_string());
		}
	}
}

int QemuGuestAdditions::execute(const std::string& command, uint32_t timeout_seconds) {
	auto timeout_chrono = std::chrono::seconds(timeout_seconds);
	coro::Timeout timeout(timeout_chrono);
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
		if (result.count("stderr")) {
			std::cout << result.at("stderr").get<std::string>();
		}
		if (result.count("stdout")) {
			std::cout << result.at("stdout").get<std::string>();
		}
		if (result.at("status").get<std::string>() == "finished") {
			return result.at("exit_code").get<int>();
		}
	}
}

void QemuGuestAdditions::copy_file_to_guest(const fs::path& src, const fs::path& dst, std::chrono::system_clock::time_point deadline) {
	std::ifstream testFile(src.generic_string(), std::ios::binary);

	std::noskipws(testFile);
	std::vector<uint8_t> fileContents = {std::istream_iterator<uint8_t>(testFile), std::istream_iterator<uint8_t>()};
	std::string encoded = base64_encode(fileContents.data(), fileContents.size());
	std::vector<uint8_t> decoded = base64_decode(encoded);

	nlohmann::json request = {
			{"method", "copy_file"},
			{"args", {
				{
					{"path", dst.generic_string()},
					{"content", encoded}
				}
			}}
	};

	if (std::chrono::system_clock::now() > (deadline - std::chrono::milliseconds(100))) {
		throw std::runtime_error("Timeout expired");
	}

	coro::Timeout timeout(deadline - std::chrono::system_clock::now());

	send(request);

	auto response = recv();

	if(!response.at("success").get<bool>()) {
		throw std::runtime_error(response.at("error").get<std::string>());
	}

}

void QemuGuestAdditions::send(const nlohmann::json& command) {
	auto command_str = command.dump();
	std::vector<uint8_t> buffer;
	uint32_t command_length = command_str.length();
	buffer.reserve(sizeof(uint32_t) + command_str.length());
	std::copy((uint8_t*)&command_length, (uint8_t*)(&command_length) + sizeof(uint32_t), std::back_inserter(buffer));

	std::copy(command_str.begin(), command_str.end(), std::back_inserter(buffer));
	socket.write(buffer);
}

nlohmann::json QemuGuestAdditions::recv() {
	uint32_t json_length = 0;
	socket.read(&json_length, sizeof(uint32_t));
	std::string json_str;
	json_str.resize(json_length);
	socket.read(&json_str[0], json_length);
	return nlohmann::json::parse(json_str);
}
