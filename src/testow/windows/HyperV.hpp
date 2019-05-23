
#pragma once

#include "../Hypervisor.hpp"
#include <hyperv/Connect.hpp>

struct HyperVGuest: Guest {
	HyperVGuest(std::string name_);

	virtual stb::Image screenshot() override;

private:
	hyperv::Connect connect;
};

struct HyperV: Hypervisor {
	HyperV();
	virtual std::vector<std::shared_ptr<Guest>> guests() const override;

private:
	wmi::CoInitializer initializer;
};
