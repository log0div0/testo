
#pragma once

#include "event.hpp"
#include "mem.hpp"
#include "kernel.hpp"

namespace cl {

struct CommandQueue {
	CommandQueue(cl_command_queue handle);
	~CommandQueue();

	CommandQueue(const CommandQueue& other) = delete;
	CommandQueue& operator=(const CommandQueue& other) = delete;
	CommandQueue(CommandQueue&& other);
	CommandQueue& operator=(CommandQueue&& other);

	cl::Event writeBuffer(cl::Mem& mem, size_t offset, size_t cb, const void* ptr, const std::vector<cl::Event>& events_wait_list = {});
	cl::Event readBuffer(cl::Mem& mem, size_t offset, size_t cb, void* ptr, const std::vector<cl::Event>& events_wait_list = {});
	cl::Event execute(cl::Kernel& kernel, const std::vector<size_t>& global_work_size, const std::vector<cl::Event>& events_wait_list = {});

private:
	cl_command_queue _handle = nullptr;
};

}
