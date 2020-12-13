
#pragma once

#include "Channel.hpp"

#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;

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

struct Server {
	Server(const std::string& fd_path);
	Server(const Server&) = delete;
	Server& operator=(const Server&) = delete;

	void run();
	void force_cancel();

private:
	bool is_canceled = false;

	void handle_command(const nlohmann::json& command);
	void handle_check_avaliable(const nlohmann::json& command);
	void handle_get_tmp_dir(const nlohmann::json& command);
	void handle_copy_file(const nlohmann::json& command);

	nlohmann::json copy_directory_out(const fs::path& dir, const fs::path& dst);
	nlohmann::json copy_single_file_out(const fs::path& dir, const fs::path& dst);

	void handle_copy_files_out(const nlohmann::json& command);
	void handle_execute(const nlohmann::json& command);

	void send_error(const std::string& error);
	Channel channel;

	VersionNumber ver;
};
