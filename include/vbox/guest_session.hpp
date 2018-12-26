
#pragma once

#include "api.hpp"
#include "progress.hpp"
#include "guest_process.hpp"
#include <Utils.hpp>
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

	bool directory_exists(const fs::path& dir, bool follow_links = false); 
	bool file_exists(const fs::path& dir, bool follow_links = false);
	void file_remove(const fs::path& path);
	void directory_create(const fs::path& dir, uint32_t mode = 0x1FD);
	void directory_remove(const fs::path& dir);
	void directory_remove_recursive(const fs::path& dir);
	Progress file_copy_to_guest(const fs::path& src, const fs::path& destination);
	GuestProcess process_create(const fs::path& executable,
		const std::vector<std::string>& arguments,
		const std::vector<std::string>& env_changes,
		const std::vector<ProcessCreateFlag>& flags,
		uint32_t timeout_ms);

	IGuestSession* handle = nullptr;
};
	
}

