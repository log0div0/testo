
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

void QemuGuestAdditions::send_raw(const uint8_t* data, size_t size) {
	size_t n = socket.write(data, size);
	if (n != size) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

void QemuGuestAdditions::recv_raw(uint8_t* data, size_t size) {
	size_t n = socket.read(data, size);
	if (n != size) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}