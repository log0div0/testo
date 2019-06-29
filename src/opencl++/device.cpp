
#include "device.hpp"
#include "error.hpp"
#include <stdexcept>

namespace cl {

Device::Device(cl_device_id id): _id(id) {}

std::string Device::name() const {
	return info(CL_DEVICE_NAME);
}

size_t Device::info_size(cl_device_info name) const {
	try {
		size_t result = 0;
		throw_if_failed(clGetDeviceInfo(_id, name, 0, nullptr, &result));
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::string Device::info(cl_device_info name) const {
	try {
		std::string result(info_size(name), '\0');
		size_t unused = 0;
		throw_if_failed(clGetDeviceInfo(_id, name, result.size(), result.data(), &unused));
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
