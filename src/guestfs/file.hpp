#include <experimental/filesystem>
#include <stdexcept>
#include <cstring>

namespace guestfs {

namespace fs = std::experimental::filesystem;

struct File {
	File() = delete;
	File(const File& other) = delete;
	File& operator=(const File& other) = delete;

	File(guestfs_h *handle_, const fs::path& path_): path(path_), handle(handle_) {
		if (!handle) {
			throw std::runtime_error(__PRETTY_FUNCTION__);
		}
		if (guestfs_touch(handle, path.generic_string().c_str()) < 0) {
			throw std::runtime_error(guestfs_last_error(handle));
		}
	}

	File(File&& other): handle(other.handle) {
		other.handle = 0;
	}

	File& operator=(File&& other) {
		std::swap(handle, other.handle);
		return *this;
	}

	size_t read(uint8_t* data, size_t size) const {
		size_t read_bytes = 0;
		auto result = guestfs_pread(handle, path.generic_string().c_str(), size, current_offset, &read_bytes);

		if (!result) {
			throw std::runtime_error(guestfs_last_error(handle));
		}

		std::memcpy((void*)data, (void*)result, read_bytes);

		delete result;

		current_offset += read_bytes;
		return read_bytes;
	}

	size_t write(const uint8_t* data, size_t size) {
		if (guestfs_write_append(handle, path.generic_string().c_str(), (char*)data, size) < 0) {
			throw std::runtime_error(guestfs_last_error(handle));
		}
		return size;
	}

	mutable int64_t current_offset = 0;

	fs::path path;
	guestfs_h *handle = nullptr;
};

}