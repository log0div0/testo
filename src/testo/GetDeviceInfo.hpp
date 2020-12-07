
#pragma once

#include <string>

struct DeviceInfo {
	std::string name;
	std::string uuid_str;
};

DeviceInfo GetDeviceInfo(int device_id);
