
#pragma once

#include <iostream>
#include <stdexcept>
#include <guestfs.h>
#include "file.hpp"

namespace guestfs {

struct Guestfs {
	Guestfs(const Guestfs& other) = delete;
	Guestfs& operator=(const Guestfs& other) = delete;
	Guestfs(const fs::path& path);
	~Guestfs();

	std::vector<std::string> list_partitions() const;

	void part_disk();
	void mkfs(const std::string& fs);
	void mount();
	void mkdir_p(const fs::path& dir);
	void upload_file(const fs::path& from, const fs::path& to);
	void upload(const fs::path& from, const fs::path& to);
	void download_file(const fs::path& from, const fs::path& to);
	void download(const fs::path& from, const fs::path& to);
	void umount();
	void touch(const fs::path& path);
	bool exists(const fs::path& path);
	bool is_file(const fs::path& path);
	bool is_dir(const fs::path& path);

	std::vector<fs::path> ls(const fs::path& dir) const;

private:
	void add_drive(const fs::path& path);
	void launch();
	void shutdown();

private:
	bool is_mounted = false;
	bool is_launched = false;
	guestfs_h *handle = nullptr;
};

}