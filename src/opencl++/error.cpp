
#include "error.hpp"
#include <stdexcept>
#include <string>

namespace cl {

void throw_if_failed(cl_int error_code) {
	if (error_code != CL_SUCCESS) {
		throw std::runtime_error("Opencl error with code " + std::to_string(error_code));
	}
}

}
