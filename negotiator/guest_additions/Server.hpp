
#pragma once

#include <string>
#include <nlohmann/json.hpp>

struct Server {
	Server(const std::string& fd_path);
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
	void handle_execute(const nlohmann::json& args);

	void send_error(const std::string& error);

	void set_interface_attribs(int speed, int parity);
	void set_blocking(bool should_block);
	int fd = 0;
	const std::string fd_path;
};
