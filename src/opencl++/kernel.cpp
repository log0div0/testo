
#include "kernel.hpp"
#include "error.hpp"
#include <stdexcept>

namespace cl {

Kernel::Kernel(cl_kernel handle): _handle(handle) {
	try {
		if (!_handle) {
			throw std::runtime_error("nullptr");
		}
	} catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Kernel::~Kernel() {
	if (_handle) {
		clReleaseKernel(_handle);
		_handle = nullptr;
	}
}

void Kernel::setArg(cl_uint index, size_t size, const void* value) {
	try {
		throw_if_failed(clSetKernelArg(_handle, index, size, value));
	} catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
