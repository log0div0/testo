
#pragma once

#include "qemu/Domain.hpp"
#include "QemuChannelHandler.hpp"
#include "../../Utils.hpp"
#include <nlohmann/json.hpp>

struct QemuGuestAdditions {
	QemuGuestAdditions() = delete;
	QemuGuestAdditions(const QemuGuestAdditions& other) = delete;
	QemuGuestAdditions& operator=(const QemuGuestAdditions& other) = delete;

	QemuGuestAdditions(vir::Domain& domain);

	bool is_avaliable();
	void copy_to_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_milliseconds);
	void copy_from_guest(const fs::path& src, const fs::path& dst, uint32_t timeout_milliseconds);
	int execute(const std::string& command, uint32_t timeout_milliseconds);

private:
	void copy_file_to_guest(const fs::path& src, const fs::path& dst, std::chrono::system_clock::time_point deadline);
	void copy_dir_to_guest(const fs::path& src, const fs::path& dst, std::chrono::system_clock::time_point deadline);

	std::shared_ptr<QemuChannelHandler> channel_handler;
};

