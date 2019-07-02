
#include "command_queue.hpp"
#include "error.hpp"
#include <stdexcept>

namespace cl {

CommandQueue::CommandQueue(cl_command_queue handle): _handle(handle) {
	try {
		if (!_handle) {
			throw std::runtime_error("nullptr");
		}
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

CommandQueue::~CommandQueue() {
	if (_handle) {
		clReleaseCommandQueue(_handle);
		_handle = nullptr;
	}
}

cl::Event CommandQueue::writeBuffer(cl::Mem& mem, size_t offset, size_t cb, const void* ptr, const std::vector<cl::Event>& events_wait_list) {
	try {
		cl_event result = nullptr;
		static_assert(sizeof(cl::Event) == sizeof(cl_event), "");
		throw_if_failed(clEnqueueWriteBuffer(_handle, mem.handle(), CL_FALSE, offset, cb, ptr, events_wait_list.size(), (const cl_event*)events_wait_list.data(), &result));
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

cl::Event CommandQueue::readBuffer(cl::Mem& mem, size_t offset, size_t cb, void* ptr, const std::vector<cl::Event>& events_wait_list) {
	try {
		cl_event result = nullptr;
		static_assert(sizeof(cl::Event) == sizeof(cl_event), "");
		throw_if_failed(clEnqueueReadBuffer(_handle, mem.handle(), CL_FALSE, offset, cb, ptr, events_wait_list.size(), (const cl_event*)events_wait_list.data(), &result));
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

cl::Event CommandQueue::execute(cl::Kernel& kernel,
	const std::vector<size_t>& global_work_size,
	const std::vector<cl::Event>& events_wait_list)
{
	try {
		cl_event result = nullptr;
		static_assert(sizeof(cl::Event) == sizeof(cl_event), "");
		throw_if_failed(clEnqueueNDRangeKernel(_handle, kernel.handle(),
			global_work_size.size(), nullptr, global_work_size.data(), nullptr,
			events_wait_list.size(), (const cl_event*)events_wait_list.data(), &result));
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

cl::Event CommandQueue::execute(cl::Kernel& kernel,
	const std::vector<size_t>& global_work_size,
	const std::vector<size_t>& local_work_size,
	const std::vector<cl::Event>& events_wait_list)
{
	try {
		if (global_work_size.size() != local_work_size.size()) {
			throw std::runtime_error("global_work_size and local_work_size have different dimensions count");
		}
		cl_event result = nullptr;
		static_assert(sizeof(cl::Event) == sizeof(cl_event), "");
		throw_if_failed(clEnqueueNDRangeKernel(_handle, kernel.handle(),
			global_work_size.size(), nullptr, global_work_size.data(), local_work_size.data(),
			events_wait_list.size(), (const cl_event*)events_wait_list.data(), &result));
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
