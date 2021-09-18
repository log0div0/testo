
#pragma once

#include "Channel.hpp"
#include "SharedFolder.hpp"
#include <testo_guest_additions_protocol/VersionNumber.hpp>

struct MessageHandler {
	MessageHandler(std::shared_ptr<Channel> channel_): channel(std::move(channel_)) {};
	~MessageHandler() = default;

	MessageHandler(const MessageHandler&) = delete;
	MessageHandler& operator=(const MessageHandler&) = delete;

	void run();

private:
	void handle_message(const nlohmann::json& command);
	void handle_check_avaliable(const nlohmann::json& command);
	void handle_get_tmp_dir(const nlohmann::json& command);
	void handle_copy_file(const nlohmann::json& command);

	nlohmann::json copy_directory_out(const fs::path& dir, const fs::path& dst);
	nlohmann::json copy_single_file_out(const fs::path& dir, const fs::path& dst);

	void handle_copy_files_out(const nlohmann::json& command);
	void handle_execute(const nlohmann::json& command);

	void handle_mount(const nlohmann::json& command);
	void handle_get_shared_folder_status(const nlohmann::json& command);
	void handle_umount(const nlohmann::json& command);

	void send_error(const std::string& error);

	std::shared_ptr<Channel> channel;

	VersionNumber ver;
};
