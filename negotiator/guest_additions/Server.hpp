
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
	nlohmann::json handle_copy_file(const nlohmann::json& args);
	nlohmann::json run(const nlohmann::json& args);

	void send_ok(const nlohmann::json& result);
	void send_error(const std::string& error);

	void set_interface_attribs(int speed, int parity);
	void set_blocking(bool should_block);
	int fd = 0;
	const std::string fd_path;
};
