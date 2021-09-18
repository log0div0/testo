
#pragma once

#include <nlohmann/json.hpp>

#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;

#include "VersionNumber.hpp"

struct GuestAdditions {
	virtual ~GuestAdditions() = default;

	bool is_avaliable(std::chrono::milliseconds timeout = std::chrono::seconds(3));
	void copy_to_guest(const fs::path& src, const fs::path& dst);
	void copy_from_guest(const fs::path& src, const fs::path& dst);
	void remove_from_guest(const fs::path& path);
	int execute(const std::string& command,
		const std::function<void(const std::string&)>& callback);
	std::string get_tmp_dir();
	bool mount(const std::string& folder_name, const fs::path& guest_path, bool permanent);
	nlohmann::json get_shared_folder_status(const std::string& folder_name);
	bool umount(const std::string& folder_name, bool permanent);

private:
	void copy_file_to_guest(const fs::path& src, const fs::path& dst);
	void copy_dir_to_guest(const fs::path& src, const fs::path& dst);

	void send(nlohmann::json command);
	nlohmann::json recv();

	VersionNumber ver;

	virtual void send_raw(const uint8_t* data, size_t size) = 0;
	virtual void recv_raw(uint8_t* data, size_t size) = 0;
};

