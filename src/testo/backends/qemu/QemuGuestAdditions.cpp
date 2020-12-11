
#include "QemuGuestAdditions.hpp"

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

size_t QemuGuestAdditions::send_raw(const uint8_t* data, size_t size) {
	return socket.write(data, size);
}

size_t QemuGuestAdditions::recv_raw(uint8_t* data, size_t size) {
	return socket.read(data, size);
}