
#include "platform.hpp"
#include "error.hpp"
#include <stdexcept>

namespace cl {

cl_uint Platform::ids_count() {
	try {
		cl_uint result = 0;
		throw_if_failed(clGetPlatformIDs(0, nullptr, &result));
		return result;
	} catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<cl_platform_id> Platform::ids() {
	try {
		std::vector<cl_platform_id> result(ids_count());
		cl_uint unused = 0;
		throw_if_failed(clGetPlatformIDs(result.size(), result.data(), &unused));
		return result;
	} catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Platform::Platform(cl_platform_id id): _id(id) {}

std::string Platform::name() const {
	return info(CL_PLATFORM_NAME);
}

cl_uint Platform::device_ids_count(cl_device_type type) const {
	try {
		cl_uint result = 0;
		throw_if_failed(clGetDeviceIDs(_id, type, 0, nullptr, &result));
		return result;
	} catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<cl_device_id> Platform::device_ids(cl_device_type type) const {
	try {
		std::vector<cl_device_id> result(device_ids_count());
		cl_uint unused = 0;
		throw_if_failed(clGetDeviceIDs(_id, type, result.size(), result.data(), &unused));
		return result;
	} catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

size_t Platform::info_size(cl_platform_info name) const {
	try {
		size_t result = 0;
		throw_if_failed(clGetPlatformInfo(_id, name, 0, nullptr, &result));
		return result;
	} catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::string Platform::info(cl_platform_info name) const {
	try {
		std::string result(info_size(name), '\0');
		size_t unused = 0;
		throw_if_failed(clGetPlatformInfo(_id, name, result.size(), result.data(), &unused));
		return result;
	} catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
