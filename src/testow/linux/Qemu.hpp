
#pragma once

#include "../Hypervisor.hpp"
#include <qemu/Host.hpp>

struct QemuGuest: Guest {
	QemuGuest(std::shared_ptr<vir::Connect> connect, vir::Domain domain);
	virtual Image screenshot() override;

private:
	std::shared_ptr<vir::Connect> connect;
	vir::Domain domain;
	std::vector<uint8_t> buffer;
};

struct Qemu: Hypervisor {
	Qemu();
	virtual std::vector<std::shared_ptr<Guest>> guests() const override;

private:
	std::shared_ptr<vir::Connect> connect;
};
