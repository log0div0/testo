
#include "../../Reporter.hpp"
#include "QemuEnvironment.hpp"
#include "QemuGuestAdditions.hpp"
#include "base64.hpp"
#include <fmt/format.h>
#include <fstream>

using namespace std::literals::chrono_literals;

QemuGuestAdditions::QemuGuestAdditions(vir::Domain& domain) {
	auto config = domain.dump_xml();
	auto devices = config.first_child().child("devices");

	for (auto channel = devices.child("channel"); channel; channel = channel.next_sibling("channel")) {
		if (std::string(channel.child("target").attribute("name").value()) == "negotiator.0") {
			if (std::string(channel.attribute("type").value()) == "unix") {
				std::string path = std::string(channel.child("source").attribute("path").value());
				channel_handler.reset(new QemuUnixChannelHandler(path));
				break;
			} else if (std::string(channel.attribute("type").value()) == "tcp") {
				auto uri = std::dynamic_pointer_cast<QemuEnvironment>(env)->uri();
				std::string begin_trimmed = uri.substr(uri.find("://") + 3);
				auto host_name = begin_trimmed.substr(0, begin_trimmed.find_first_of(":/"));
				auto port = channel.child("source").attribute("service").value();
				channel_handler.reset(new QemuTCPChannelHandler(host_name, port));
				break;
			} else {
				throw std::runtime_error("Unknown channel type: " + std::string(channel.attribute("type").value()));
			}
		}
	}

	if (!channel_handler) {
		throw std::runtime_error("Can't find negotiator target in vm config");
	}
}

bool QemuGuestAdditions::is_avaliable() {
	try {
		nlohmann::json request = {
			{"method", "check_avaliable"}
		};

		coro::Timeout timeout(3s);

		channel_handler->send(request);

		auto response = channel_handler->recv();
		return response.at("success").get<bool>();
	} catch (const std::exception& error) {
		std::cout << "Checking guest additions avaliability failed: " << error.what() << std::endl;
		return false;
	}
}

void QemuGuestAdditions::copy_to_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_milliseconds) {
	auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_milliseconds);
	//4) Now we're all set
	if (fs::is_regular_file(src)) {
		copy_file_to_guest(src, dst, deadline);
	} else if (fs::is_directory(src)) {
		copy_dir_to_guest(src, dst, deadline);
	} else {
		throw std::runtime_error("Unknown type of file: " + src.generic_string());
	}
}

void QemuGuestAdditions::copy_from_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_milliseconds) {
	nlohmann::json request = {
		{"method", "copy_files_out"}
	};

	request["args"] = nlohmann::json::array();
	request["args"].push_back(src.generic_string());
	request["args"].push_back(dst.generic_string());

	auto chrono_seconds = std::chrono::milliseconds(timeout_milliseconds);
	coro::Timeout timeout(chrono_seconds); //actually, it really depends on file size, TODO

	channel_handler->send(request);

	auto response = channel_handler->recv();

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

int QemuGuestAdditions::execute(const std::string& command, uint32_t timeout_milliseconds) {
	auto timeout_chrono = std::chrono::milliseconds(timeout_milliseconds);
	coro::Timeout timeout(timeout_chrono);
	nlohmann::json request = {
			{"method", "execute"},
			{"args", {
				command
			}}
	};

	channel_handler->send(request);

	while (true) {
		auto response = channel_handler->recv();
		if (!response.at("success").get<bool>()) {
			throw std::runtime_error(std::string("Negotiator inner error: ") + response.at("error").get<std::string>());
		}

		auto result = response.at("result");
		if (result.count("stderr")) {
			reporter.exec_command_output(result.at("stderr").get<std::string>());
		}
		if (result.count("stdout")) {
			reporter.exec_command_output(result.at("stdout").get<std::string>());
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

	nlohmann::json request = {
			{"method", "copy_file"},
			{"args", {
				{
					{"path", dst.generic_string()},
					{"content", encoded}
				}
			}}
	};

	coro::Timeout timeout(deadline - std::chrono::system_clock::now());

	channel_handler->send(request);

	auto response = channel_handler->recv();

	if(!response.at("success").get<bool>()) {
		throw std::runtime_error(response.at("error").get<std::string>());
	}

}
