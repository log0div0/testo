
#pragma once

#include "api.hpp"
#include <vector>

namespace vbox {

struct GuestProcess {
	GuestProcess(IGuestProcess* handle);
	virtual ~GuestProcess();

	GuestProcess(const GuestProcess&) = delete;
	GuestProcess& operator=(const GuestProcess&) = delete;
	GuestProcess(GuestProcess&& other);
	GuestProcess& operator=(GuestProcess&& other);

	ProcessWaitResult wait_for(ProcessWaitForFlag wait_flag, uint32_t timeout_ms);
	ProcessWaitResult wait_for(const std::vector<ProcessWaitForFlag>& flags, uint32_t timeout_ms);

	std::vector<uint8_t> read(uint32_t handle, uint32_t bytes_to_read, uint32_t timeout_ms);

	LONG exit_code() const;
	ProcessStatus status() const;
	bool is_alive() const;

	IGuestProcess* handle = nullptr;
};

}
