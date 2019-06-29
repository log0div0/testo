
#pragma once

#include "device.hpp"

namespace cl {

struct Platform {
	static cl_uint ids_count();
	static std::vector<cl_platform_id> ids();

	Platform(cl_platform_id id);

	std::string name() const;

	cl_uint device_ids_count(cl_device_type type = CL_DEVICE_TYPE_ALL) const;
	std::vector<cl_device_id> device_ids(cl_device_type type = CL_DEVICE_TYPE_ALL) const;

private:
	size_t info_size(cl_platform_info name) const;
	std::string info(cl_platform_info name) const;

	cl_platform_id _id;
};

}
