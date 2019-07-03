
#include "mem.hpp"
#include "error.hpp"
#include <stdexcept>

namespace cl {

Mem::Mem(cl_mem handle): _handle(handle) {
	try {
		if (!_handle) {
			throw std::runtime_error("nullptr");
		}
	} catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Mem::~Mem() {
	if (_handle) {
		clReleaseMemObject(_handle);
		_handle = nullptr;
	}
}

}
