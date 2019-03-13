
#include "Negotiator.hpp"
#include "base64.hpp"
#include <fmt/format.h>
#include <fstream>

using namespace std::literals::chrono_literals;

Negotiator::Negotiator(vir::Domain& domain) {
	endpoint = Endpoint(fmt::format("/var/lib/libvirt/qemu/channel/target/domain-{}-{}/negotiator.0",
		domain.id(), domain.name()));

	socket.connect(endpoint);
}

bool Negotiator::is_avaliable() {
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

void Negotiator::copy_to_guest(const fs::path& src, const fs::path& dst) {
	//4) Now we're all set
	if (fs::is_regular_file(src)) {
		copy_file_to_guest(src, dst);
	} else if (fs::is_directory(src)) {
		copy_dir_to_guest(src, dst);
	} else {
		throw std::runtime_error("Unknown type of file: " + src.generic_string());
	}
}

void Negotiator::copy_dir_to_guest(const fs::path& src, const fs::path& dst) {
	for (auto& file: fs::directory_iterator(src)) {
		if (fs::is_regular_file(file)) {
			copy_file_to_guest(file, dst / file.path().filename());
		} else if (fs::is_directory(file)) {
			copy_dir_to_guest(file, dst / file.path().filename());
		} //else continue
	}
}

void Negotiator::copy_file_to_guest(const fs::path& src, const fs::path& dst) {
	std::ifstream testFile(src.generic_string(), std::ios::binary);
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

	coro::Timeout timeout(10s); //actually, it really depends on file size, TODO

	send(request);

	auto response = recv();

	if(!response.at("success").get<bool>()) {
		throw std::runtime_error(response.at("error").get<std::string>());
	}

}

void Negotiator::send(const nlohmann::json& command) {
	auto command_str = command.dump();
	std::vector<uint8_t> buffer;
	uint32_t command_length = command_str.length();
	buffer.reserve(sizeof(uint32_t) + command_str.length());
	std::copy((uint8_t*)&command_length, (uint8_t*)(&command_length) + sizeof(uint32_t), std::back_inserter(buffer));

	std::copy(command_str.begin(), command_str.end(), std::back_inserter(buffer));
	socket.write(buffer);
}

nlohmann::json Negotiator::recv() {
	uint32_t json_length = 0;
	socket.read(&json_length, sizeof(uint32_t));
	std::string json_str;
	json_str.resize(json_length);
	socket.read(&json_str[0], json_length);
	return nlohmann::json::parse(json_str);
}
