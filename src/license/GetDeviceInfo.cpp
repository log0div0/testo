
#include "GetDeviceInfo.hpp"
#include <stdexcept>

#include <cuda_runtime.h>

char hex_map[] = {
	'0',
	'1',
	'2',
	'3',
	'4',
	'5',
	'6',
	'7',
	'8',
	'9',
	'a',
	'b',
	'c',
	'd',
	'e',
	'f',
};

DeviceInfo GetDeviceInfo(int device_id) {
	cudaDeviceProp props = {};
	cudaError_t error = cudaDeviceSynchronize();
	if (error != cudaSuccess) {
		throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") + cudaGetErrorString(error));
	}
	if (cudaGetDeviceProperties(&props, device_id) != cudaSuccess) {
		throw std::runtime_error("cudaGetDeviceProperties failed");
	}
	DeviceInfo info;
	info.name = props.name;
	size_t i = 0;
	uint8_t* uuid = (uint8_t*)&(props.uuid);
	while (i < 4) {
		info.uuid_str += hex_map[uuid[i] >> 4];
		info.uuid_str += hex_map[uuid[i] & 0xf];
		++i;
	}
	info.uuid_str += '-';
	while (i < 6) {
		info.uuid_str += hex_map[uuid[i] >> 4];
		info.uuid_str += hex_map[uuid[i] & 0xf];
		++i;
	}
	info.uuid_str += '-';
	while (i < 8) {
		info.uuid_str += hex_map[uuid[i] >> 4];
		info.uuid_str += hex_map[uuid[i] & 0xf];
		++i;
	}
	info.uuid_str += '-';
	while (i < 10) {
		info.uuid_str += hex_map[uuid[i] >> 4];
		info.uuid_str += hex_map[uuid[i] & 0xf];
		++i;
	}
	info.uuid_str += '-';
	while (i < 16) {
		info.uuid_str += hex_map[uuid[i] >> 4];
		info.uuid_str += hex_map[uuid[i] & 0xf];
		++i;
	}
	return info;
}
