
#pragma once

#include <string>
#include <nlohmann/json.hpp>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

struct Server {
	Server(const fs::path& fd_path);
	~Server();
	Server(const Server&) = delete;
	Server& operator=(const Server&) = delete;

	void run();

private:
	nlohmann::json read();
	void send(const nlohmann::json& response);

	void handle_command(const nlohmann::json& command);
	void handle_check_avaliable();
	void handle_copy_file(const nlohmann::json& args);

	nlohmann::json copy_directory(const fs::path& dir, const fs::path& dst);
	nlohmann::json copy_single_file(const fs::path& dir, const fs::path& dst);

	void handle_copy_files_out(const nlohmann::json& args);
	void handle_execute(const nlohmann::json& args);

	void send_error(const std::string& error);
	int fd = 0;
	const fs::path fd_path;
};
