
#pragma once

#include <CL/cl.h>
#include <vector>
#include <string>

namespace cl {

struct Device {
	Device(cl_device_id id);

	std::string name() const;
	cl_device_id id() const { return _id; }

private:
	size_t info_size(cl_device_info name) const;
	std::string info(cl_device_info name) const;

	cl_device_id _id;
};

}
