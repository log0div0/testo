
#include "device.hpp"
#include "error.hpp"
#include <stdexcept>

namespace cl {

Device::Device(cl_device_id id): _id(id) {}

std::string Device::name() const {
	return info_str(CL_DEVICE_NAME);
}

size_t Device::max_work_group_size() const {
	return info_pod<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE);
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

std::vector<uint8_t> Device::info(cl_device_info name) const {
	try {
		std::vector<uint8_t> result(info_size(name), 0);
		size_t unused = 0;
		throw_if_failed(clGetDeviceInfo(_id, name, result.size(), result.data(), &unused));
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::string Device::info_str(cl_device_info name) const {
	try {
		std::vector<uint8_t> buffer = info(name);
		return std::string((char*)buffer.data(), buffer.size());
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

template <typename POD>
POD Device::info_pod(cl_device_info name) const {
	try {
		std::vector<uint8_t> buffer = info(name);
		if (buffer.size() != sizeof(POD)) {
			throw std::runtime_error("Size mismatch");
		}
		return *(POD*)buffer.data();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
