
#include "context.hpp"
#include "error.hpp"
#include <stdexcept>

namespace cl {

Context::Context(Platform platform, std::vector<Device> devices) {
	try {
		cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform.id(), 0};
		cl_int error_code = 0;
		static_assert(sizeof(cl_device_id) == sizeof(Device));
		_handle = clCreateContext(properties, devices.size(), (cl_device_id*)devices.data(), nullptr, nullptr, &error_code);
		throw_if_failed(error_code);
	} catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Context::~Context() {
	if (_handle) {
		clReleaseContext(_handle);
		_handle = nullptr;
	}
}

Context::Context(Context&& other): _handle(other._handle) {
	other._handle = nullptr;
}

Context& Context::operator=(Context&& other) {
	std::swap(_handle, other._handle);
	return *this;
}

CommandQueue Context::createCommandQueue(Device& device) {
	try {
		cl_int error_code = 0;
		cl_command_queue result = clCreateCommandQueue(_handle, device.id(), 0, &error_code);
		throw_if_failed(error_code);
		return result;
	} catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Program Context::createProgram(const std::vector<std::string>& sources) {
	try {
		cl_int error_code = 0;
		std::vector<const char*> strings;
		for (auto& source: sources) {
			strings.push_back(source.c_str());
		}
		cl_program result = clCreateProgramWithSource(_handle, strings.size(), strings.data(), nullptr, &error_code);
		throw_if_failed(error_code);
		return result;
	} catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Mem Context::createBuffer(cl_mem_flags flags, size_t size, void* host_ptr) {
	try {
		cl_int error_code = 0;
		cl_mem result = clCreateBuffer(_handle, flags, size, host_ptr, &error_code);
		throw_if_failed(error_code);
		return result;
	} catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
