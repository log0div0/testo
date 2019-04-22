
#pragma once

#include "api.hpp"
#include "progress.hpp"
#include "guest_process.hpp"
#include <vector>

namespace vbox  {

struct GuestSession {
	GuestSession(IGuestSession* handle);
	~GuestSession();

	GuestSession(const GuestSession& other) = delete;
	GuestSession& operator=(const GuestSession& other) = delete;

	GuestSession(GuestSession&& other);
	GuestSession& operator=(GuestSession&& other);

	GuestSessionWaitResult wait_for(GuestSessionWaitForFlag wait_flag, uint32_t timeout_ms);
	void close();

	bool directory_exists(const std::string& dir, bool follow_links = false);
	bool file_exists(const std::string& dir, bool follow_links = false);
	void file_remove(const std::string& path);
	void directory_create(const std::string& dir, uint32_t mode = 0x1FD);
	void directory_remove(const std::string& dir);
	void directory_remove_recursive(const std::string& dir);
	Progress file_copy_to_guest(const std::string& src, const std::string& destination);
	GuestProcess process_create(const std::string& executable,
		const std::vector<std::string>& arguments,
		const std::vector<std::string>& env_changes,
		const std::vector<ProcessCreateFlag>& flags,
		uint32_t timeout_ms);

	IGuestSession* handle = nullptr;
};

}

