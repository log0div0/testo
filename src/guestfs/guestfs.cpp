
#include "guestfs.hpp"

#ifdef WIN32
#include "../winapi.hpp"
#else
#include "../linuxapi.hpp"
#endif

#include "coro/CheckPoint.h"

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

void Guestfs::upload_file(const fs::path& from, const fs::path& to) {
#ifdef WIN32
	winapi::File source(from.generic_string(), GENERIC_READ, OPEN_EXISTING);
#else
	linuxapi::File source(from, O_RDONLY, 0);
#endif

	File dest(handle, to);
	uint8_t buf[8192];
	size_t size;
	while ((size = source.read(buf, sizeof(buf))) > 0) {
		dest.write(buf, size);
		coro::CheckPoint();
	}
}

void Guestfs::upload(const fs::path& from, const fs::path& to) {
	if (!fs::exists(from)) {
		throw std::runtime_error("upload error: \"from\" path " + from.generic_string() + " does not exist");
	}

	if (!is_file(to) && !is_dir(to) && exists(to)) {
		throw std::runtime_error("Fs_copy: Unsupported type of destination: " + to.generic_string());
	}

	if (fs::is_directory(from) && is_file(to)) {
		throw std::runtime_error("Fs_copy: can't copy a directory " + from.generic_string() + " to a regular file " + to.generic_string());
	}

	//if from is a regular file
	if (fs::is_regular_file(from)) {
		if (is_dir(to)) {
			upload_file(from, to / from.filename());
		} else {
			mkdir_p(to.parent_path());
			upload_file(from, to);
		}
	} else if (fs::is_directory(from)) {
		mkdir_p(to);
		for (auto& directory_entry: fs::directory_iterator(from)) {
			upload(directory_entry.path(), to / directory_entry.path().filename());
		}
	} else {
		throw std::runtime_error("Fs_copy: Unsupported type of file: " + from.generic_string());
	}
}

void Guestfs::download(const fs::path& from, const fs::path& to) {
	if (!exists(from)) {
		throw std::runtime_error("download error: \"from\" path " + from.generic_string() + " does not exist");
	}

	if (!is_file(to) && !is_dir(to) && exists(to)) {
		throw std::runtime_error("Fs_copy: Unsupported type of destination: " + to.generic_string());
	}

	if (is_dir(from) && fs::is_regular_file(to)) {
		throw std::runtime_error("Fs_copy: can't copy a directory " + from.generic_string() + " to a regular file " + to.generic_string());
	}

	//if from is a regular file
	if (is_file(from)) {
		if (fs::is_directory(to)) {
			download_file(from, to / from.filename());
		} else {
			fs::create_directories(to.parent_path());
			download_file(from, to);
		}
	} else if (is_dir(from)) {
		fs::create_directories(to);

		for (auto& directory_entry: ls(from)) {
			download(directory_entry, to / directory_entry.filename());
		}
	} else {
		throw std::runtime_error("Fs_copy: Unsupported type of file: " + from.generic_string());
	}
}

void Guestfs::download_file(const fs::path& from, const fs::path& to) {
	File source(handle, from);
#ifdef WIN32
	winapi::File dest(to.generic_string(), GENERIC_WRITE, CREATE_ALWAYS);
#else
	linuxapi::File dest(to, O_WRONLY | O_CREAT, 0644);
#endif

	uint8_t buf[8192];
	size_t size;
	while ((size = source.read(buf, sizeof(buf))) > 0) {
		dest.write(buf, size);
		coro::CheckPoint();
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

bool Guestfs::exists(const fs::path& path) {
	int result = guestfs_exists(handle, path.generic_string().c_str());
	if (result < 0) {
		throw std::runtime_error(guestfs_last_error(handle));
	}
	return result;
}

bool Guestfs::is_file(const fs::path& path) {
	int result = guestfs_is_file(handle, path.generic_string().c_str());
	if (result < 0) {
		throw std::runtime_error(guestfs_last_error(handle));
	}
	return result;
}

bool Guestfs::is_dir(const fs::path& path) {
	int result = guestfs_is_dir(handle, path.generic_string().c_str());
	if (result < 0) {
		throw std::runtime_error(guestfs_last_error(handle));
	}
	return result;
}

std::vector<fs::path> Guestfs::ls(const fs::path& dir) const {
	char** ls_result = guestfs_ls(handle, dir.generic_string().c_str());

	if (!ls_result) {
		throw std::runtime_error(guestfs_last_error(handle));
	}

	std::vector<fs::path> result;

	for (size_t i = 0; ls_result[i] != nullptr; i++) {
		result.push_back(dir / fs::path(ls_result[i]));
		free(ls_result[i]);
	}

	free(ls_result);
	return result;
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