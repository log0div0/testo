
#pragma once

#include <nlohmann/json.hpp>
#include "../Utils.hpp"

struct GuestAdditions {
	virtual ~GuestAdditions() = default;

	bool is_avaliable();
	void copy_to_guest(const fs::path& src, const fs::path& dst);
	void copy_from_guest(const fs::path& src, const fs::path& dst);
	int execute(const std::string& command,
		const std::function<void(const std::string&)>& callback);
	std::string get_tmp_dir();

private:
	void copy_file_to_guest(const fs::path& src, const fs::path& dst);
	void copy_dir_to_guest(const fs::path& src, const fs::path& dst);

	void send(const nlohmann::json& command);
	nlohmann::json recv();

	virtual size_t send_raw(const uint8_t* data, size_t size) = 0;
	virtual size_t recv_raw(uint8_t* data, size_t size) = 0;
};

