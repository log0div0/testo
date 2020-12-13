
#pragma once

#ifdef WIN32
#include <winapi/File.hpp>
#else
#include <posixapi/File.hpp>
#endif

#include <vector>

namespace os {

#ifdef WIN32
struct File: winapi::File {
	using winapi::File::File;
#else
struct File: posixapi::File {
	using posixapi::File::File;
#endif
	static File open_for_write(const std::string& path);
	static File open_for_read(const std::string& path);

	File(File&& other);
	File& operator=(File&& other);

	uint64_t size();
	std::vector<uint8_t> read_all();
};

}
