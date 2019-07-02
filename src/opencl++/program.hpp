
#pragma once

#include "device.hpp"
#include "kernel.hpp"
#include <vector>
#include <string>

namespace cl {

struct Program {
	Program() = default;
	Program(cl_program handle);
	~Program();

	Program(const Program& other) = delete;
	Program& operator=(const Program& other) = delete;
	Program(Program&& other);
	Program& operator=(Program&& other);

	void build(const std::vector<Device>& devices, const std::string& options = {});

	std::string build_log(Device device) const;

	Kernel createKernel(const std::string& name);

private:
	size_t build_info_size(Device device, cl_program_build_info name) const;
	std::string build_info(Device device, cl_program_build_info name) const;

	cl_program _handle = nullptr;
};

}
