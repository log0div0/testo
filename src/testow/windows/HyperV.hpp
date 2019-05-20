
#pragma once

#include "../Hypervisor.hpp"
#include <hyperv/Connect.hpp>

struct HyperVGuest: Guest {
	HyperVGuest(hyperv::Machine machine_);

	virtual bool is_running() const override;
	virtual stb::Image screenshot() const override;

private:
	hyperv::Machine machine;
};

struct HyperV: Hypervisor {
	HyperV();
	virtual std::vector<std::shared_ptr<Guest>> guests() const override;

private:
	wmi::CoInitializer initializer;
	std::unique_ptr<hyperv::Connect> connect;
};
