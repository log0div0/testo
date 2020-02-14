
#pragma once

#include "qemu/Domain.hpp"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic push
#include <coro/StreamSocket.h>
#include <coro/Timeout.h>
#pragma GCC diagnostic pop
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
	std::string get_tmp_dir();

private:
	using Socket = coro::StreamSocket<asio::local::stream_protocol>;
	using Endpoint = asio::local::stream_protocol::endpoint;

	void copy_file_to_guest(const fs::path& src, const fs::path& dst, std::chrono::system_clock::time_point deadline);
	void copy_dir_to_guest(const fs::path& src, const fs::path& dst, std::chrono::system_clock::time_point deadline);

	void send(const nlohmann::json& command);
	nlohmann::json recv();

	Socket socket;
	Endpoint endpoint;
};

