
#pragma once

#include "Channel.hpp"

struct Server {
	Server(const fs::path& fd_path);
	Server(const Server&) = delete;
	Server& operator=(const Server&) = delete;

	void run();

private:

	void handle_command(const nlohmann::json& command);
	void handle_check_avaliable();
	void handle_copy_file(const nlohmann::json& args);

	nlohmann::json copy_directory_out(const fs::path& dir, const fs::path& dst);
	nlohmann::json copy_single_file_out(const fs::path& dir, const fs::path& dst);

	void handle_copy_files_out(const nlohmann::json& args);
	void handle_execute(const nlohmann::json& args);

	void send_error(const std::string& error);
	Channel channel;
	const fs::path fd_path;
};
