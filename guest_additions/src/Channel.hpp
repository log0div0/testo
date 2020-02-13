
#pragma once

#include <nlohmann/json.hpp>
#ifdef __GNUC__
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

struct Channel {
	Channel() = default;
	Channel(const fs::path& fd_path);
	~Channel();

	Channel(Channel&& other);
	Channel& operator=(Channel&& other);

	nlohmann::json read();
	void send(const nlohmann::json& response);

	int fd = -1;
};
