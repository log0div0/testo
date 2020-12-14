
#include "File.hpp"

namespace os {

File File::open_for_read(const std::string& path) {
#ifdef WIN32
	return File(path, GENERIC_READ, OPEN_EXISTING);
#else
	return File(path, O_RDONLY, 0);
#endif
}

File File::open_for_write(const std::string& path) {
#ifdef WIN32
	return File(path, GENERIC_WRITE, CREATE_ALWAYS);
#else
	return File(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
#endif
}

uint64_t File::size() {
#ifdef WIN32
	return winapi::File::size();
#else
	return posixapi::File::get_stat().st_size;
#endif
}

std::vector<uint8_t> File::read_all() {
	std::vector<uint8_t> buf(size());
	read(buf.data(), buf.size());
	return buf;
}

}
