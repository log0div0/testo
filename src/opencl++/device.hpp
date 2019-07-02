
#pragma once

#include <CL/cl.h>
#include <vector>
#include <string>

namespace cl {

struct Device {
	Device() = default;
	Device(cl_device_id id);

	std::string name() const;
	size_t max_work_group_size() const;
	cl_device_id id() const { return _id; }

private:
	size_t info_size(cl_device_info name) const;
	std::vector<uint8_t> info(cl_device_info name) const;
	std::string info_str(cl_device_info name) const;
	template <typename POD>
	POD info_pod(cl_device_info name) const;

	cl_device_id _id = nullptr;
};

}
