
#pragma once

#include <CL/cl.h>

namespace cl {

struct Mem {
	Mem(cl_mem handle);
	~Mem();

	Mem(const Mem& other) = delete;
	Mem& operator=(const Mem& other) = delete;
	Mem(Mem&& other);
	Mem& operator=(Mem&& other);

	cl_mem handle() const { return _handle; }

private:
	cl_mem _handle = nullptr;
};

}
