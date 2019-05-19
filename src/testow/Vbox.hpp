
#pragma once

#include "Hypervisor.hpp"
#include <vbox/virtual_box_client.hpp>
#include <vbox/virtual_box.hpp>

struct VboxGuest: Guest {
	VboxGuest(vbox::Machine machine_, vbox::Session session_);
	~VboxGuest();
	virtual bool is_running() const override;
	virtual stb::Image screenshot() const override;

private:
	vbox::Machine machine;
	vbox::Session session;
};

struct Vbox: Hypervisor {
	Vbox();
	virtual std::vector<std::shared_ptr<Guest>> guests() const override;

private:
	vbox::API api;
	vbox::VirtualBoxClient virtual_box_client;
	vbox::VirtualBox virtual_box;
};