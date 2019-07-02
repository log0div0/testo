
#include "program.hpp"
#include "error.hpp"
#include <stdexcept>

namespace cl {

Program::Program(cl_program handle): _handle(handle) {
	try {
		if (!_handle) {
			throw std::runtime_error("nullptr");
		}
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Program::~Program() {
	if (_handle) {
		clReleaseProgram(_handle);
		_handle = nullptr;
	}
}

Program::Program(Program&& other): _handle(other._handle) {
	other._handle = nullptr;
}

Program& Program::operator=(Program&& other) {
	std::swap(_handle, other._handle);
	return *this;
}

void Program::build(const std::vector<Device>& devices, const std::string& options) {
	try {
		static_assert(sizeof(cl_device_id) == sizeof(Device));
		throw_if_failed(clBuildProgram(_handle, devices.size(), (cl_device_id*)devices.data(), options.c_str(), nullptr, nullptr));
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::string Program::build_log(Device device) const {
	try {
		return build_info(device, CL_PROGRAM_BUILD_LOG);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Kernel Program::createKernel(const std::string& name) {
	try {
		cl_int error_code = 0;
		cl_kernel result = clCreateKernel(_handle, name.c_str(), &error_code);
		throw_if_failed(error_code);
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

size_t Program::build_info_size(Device device, cl_program_build_info name) const {
	try {
		size_t result = 0;
		throw_if_failed(clGetProgramBuildInfo(_handle, device.id(), name, 0, nullptr, &result));
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::string Program::build_info(Device device, cl_program_build_info name) const {
	try {
		std::string result(build_info_size(device, name), '\0');
		size_t unused = 0;
		throw_if_failed(clGetProgramBuildInfo(_handle, device.id(), name, result.size(), result.data(), &unused));
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
