
#pragma once

#include "command_queue.hpp"
#include "program.hpp"
#include "platform.hpp"
#include "mem.hpp"

namespace cl {

struct Context {
	Context(Platform platform, std::vector<Device> devices);
	~Context();

	Context(const Context& other) = delete;
	Context& operator=(const Context& other) = delete;
	Context(Context&& other);
	Context& operator=(Context&& other);

	cl_context handle() const { return _handle; }

	CommandQueue createCommandQueue(Device& device);
	Program createProgram(const std::vector<std::string>& sources);
	Mem createBuffer(cl_mem_flags flags, size_t size, void* host_ptr = nullptr);

private:
	cl_context _handle = nullptr;
};

}
