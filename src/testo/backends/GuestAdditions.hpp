
#pragma once

#include <nlohmann/json.hpp>
#include "../Utils.hpp"

struct VersionNumber {
	int MAJOR = 0;
	int MINOR = 0;
	int PATCH = 0;

	VersionNumber() = default;
	VersionNumber(int a, int b, int c):
		MAJOR(a),
		MINOR(b),
		PATCH(c) {}
	VersionNumber(const std::string& str);

	bool operator<(const VersionNumber& other);

	std::string to_string() const;
};

struct GuestAdditions {
	virtual ~GuestAdditions() = default;

	bool is_avaliable();
	void copy_to_guest(const fs::path& src, const fs::path& dst);
	void copy_from_guest(const fs::path& src, const fs::path& dst);
	void remove_from_guest(const fs::path& path);
	int execute(const std::string& command,
		const std::function<void(const std::string&)>& callback);
	std::string get_tmp_dir();

private:
	void copy_file_to_guest(const fs::path& src, const fs::path& dst);
	void copy_dir_to_guest(const fs::path& src, const fs::path& dst);

	void send(nlohmann::json command);
	nlohmann::json recv();

	VersionNumber ver;

	virtual void send_raw(const uint8_t* data, size_t size) = 0;
	virtual void recv_raw(uint8_t* data, size_t size) = 0;
};

