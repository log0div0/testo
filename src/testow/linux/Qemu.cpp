
#include "Qemu.hpp"

QemuGuest::QemuGuest(std::shared_ptr<vir::Connect> connect_, vir::Domain domain_): Guest(domain_.name()),
	connect(std::move(connect_)),
	domain(std::move(domain_)) {}

Image QemuGuest::screenshot() {
	if (!domain.is_active()) {
		return {};
	}
	if (!buffer.size()) {
		buffer.resize(10'000'000);
	}
	auto stream = connect->new_stream();
	auto mime = domain.screenshot(stream);
	size_t bytes = stream.recv_all(buffer.data(), buffer.size());
	stream.finish();

	return Image(buffer.data(), bytes);
}

Qemu::Qemu() {
	connect = std::make_shared<vir::Connect>(vir::connect_open("qemu:///system"));
}

std::vector<std::shared_ptr<Guest>> Qemu::guests() const {
	std::vector<std::shared_ptr<Guest>> result;
	for (auto& domain: connect->domains({VIR_CONNECT_LIST_DOMAINS_PERSISTENT})) {
		result.push_back(std::make_shared<QemuGuest>(connect, std::move(domain)));
	}
	return result;
}
