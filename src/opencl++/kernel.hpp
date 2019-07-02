
#pragma once

#include <CL/cl.h>

namespace cl {

struct Kernel {
	Kernel() = default;
	Kernel(cl_kernel handle);
	~Kernel();

	Kernel(const Kernel& other) = delete;
	Kernel& operator=(const Kernel& other) = delete;
	Kernel(Kernel&& other);
	Kernel& operator=(Kernel&& other);

	cl_kernel handle() const { return _handle; }

	void setArg(cl_uint index, size_t size, const void* value);

private:
	cl_kernel _handle = nullptr;
};

}
