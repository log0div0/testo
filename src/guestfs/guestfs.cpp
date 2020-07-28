
#include "guestfs.hpp"

namespace guestfs {

Guestfs::Guestfs(const fs::path& path) {
	handle = guestfs_create();
	if (!handle) {
		throw std::runtime_error("init guestfs");
	}

	guestfs_set_error_handler(handle, NULL, NULL);
	add_drive(path);
	launch();
}


Guestfs::~Guestfs() {
	if (is_mounted) {
		umount();
	}
	if (is_launched) {
		shutdown();
	}
	if (handle) {
		guestfs_close(handle);
	}
}

std::vector<std::string> Guestfs::list_partitions() const {
	std::vector<std::string> result;
	char** partitions = guestfs_list_partitions(handle);
	if (!partitions) {
		throw std::runtime_error(guestfs_last_error(handle));
	}

	for (size_t i = 0; partitions[i] != nullptr; i++) {
		result.push_back(partitions[i]);
		free(partitions[i]);
	}

	free(partitions);
	return result;
}

File Guestfs::file(const fs::path& path) {
	return File(handle, path);
}

void Guestfs::part_disk() {
	if (guestfs_part_disk(handle, "/dev/sda", "mbr") < 0) {
		throw std::runtime_error(guestfs_last_error(handle));
	}
}

void Guestfs::mkfs(const std::string& fs) {
	auto partitions = list_partitions();
	if (!partitions.size()) {
		throw std::runtime_error("There's no partition to mkfs");
	}

	if (guestfs_mkfs(handle, fs.c_str(), partitions[0].c_str()) < 0) {
		throw std::runtime_error(guestfs_last_error(handle));
	}
}

void Guestfs::mount() {
	auto partitions = list_partitions();
	if (!partitions.size()) {
		throw std::runtime_error("There's no partition to mount");
	}

	if (guestfs_mount(handle, partitions[0].c_str(), "/") < 0) {
		throw std::runtime_error(guestfs_last_error(handle));
	}
	is_mounted = true;
}

void Guestfs::mkdir_p(const fs::path& dir) {
	if (guestfs_mkdir_p(handle, dir.generic_string().c_str()) < 0) {
		throw std::runtime_error(guestfs_last_error(handle));
	}
}

void Guestfs::upload(const fs::path& from, const fs::path& to) {
	if (guestfs_upload(handle, from.generic_string().c_str(), to.generic_string().c_str()) < 0) {
		throw std::runtime_error(guestfs_last_error(handle));
	}
}

void Guestfs::umount() {
	if (guestfs_umount(handle, "/") < 0) {
		throw std::runtime_error(guestfs_last_error(handle));
	}
	is_mounted = false;
}

void Guestfs::touch(const fs::path& path) {
	if (guestfs_touch(handle, path.generic_string().c_str())) {
		throw std::runtime_error(guestfs_last_error(handle));
	}
}

void Guestfs::add_drive(const fs::path& path) {
	if (guestfs_add_drive(handle, path.generic_string().c_str()) < 0) {
		throw std::runtime_error(guestfs_last_error(handle));
	}
}

void Guestfs::launch() {
	if (is_launched) {
		throw std::runtime_error("Guestfs is already launched");
	}
	if (guestfs_launch(handle) < 0) {
		throw std::runtime_error(guestfs_last_error(handle));
	}
	is_launched = true;
}

void Guestfs::shutdown() {
	if (guestfs_shutdown(handle) < 0) {
		throw std::runtime_error(guestfs_last_error(handle));
	}
	is_launched = false;
}


}