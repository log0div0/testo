
#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <string>

namespace posixapi {

struct File {
	File() = default;
	File(const std::string& path, int flags, int mode);
	~File();

	File(File&& other);
	File& operator=(File&& other);

	size_t read(uint8_t* data, size_t size);
	size_t write(const uint8_t* data, size_t size);

	struct stat get_stat();

	int handle = -1;
};

}
