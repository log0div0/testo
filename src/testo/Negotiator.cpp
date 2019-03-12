
#include "Negotiator.hpp"
#include <fmt/format.h>

using namespace std::literals::chrono_literals;

Negotiator::Negotiator(vir::Domain& domain) {
	endpoint = Endpoint(fmt::format("/var/lib/libvirt/qemu/channel/target/domain-{}-{}/negotiator.0",
		domain.id(), domain.name()));
}

bool Negotiator::is_avaliable() {
	try {
		socket.connect(endpoint);

		nlohmann::json request = {
			{"method", "check_avaliable"}
		};

		send(request);

		auto response = recv();
		return response.at("success").get<bool>();

	} catch (...) {
		return false;
	}
}

void Negotiator::send(const nlohmann::json& command) {
	coro::Timeout timeout(3s);
	auto command_str = command.dump();
	std::vector<uint8_t> buffer;
	uint32_t command_length = command_str.length();
	buffer.reserve(sizeof(uint32_t) + command_str.length());
	std::copy((uint8_t*)&command_length, (uint8_t*)(&command_length) + sizeof(uint32_t), std::back_inserter(buffer));

	std::copy(command_str.begin(), command_str.end(), std::back_inserter(buffer));
	socket.write(buffer);
}

nlohmann::json Negotiator::recv() {
	coro::Timeout timeout(15s); //for now
	uint32_t json_length = 0;
	socket.read(&json_length, sizeof(uint32_t));
	std::string json_str;
	json_str.resize(json_length);
	socket.read(&json_str[0], json_length);
	return nlohmann::json::parse(json_str);
}
