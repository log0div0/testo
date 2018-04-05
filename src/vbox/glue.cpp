
#include "glue.hpp"
#include <stdexcept>

namespace vbox {

Glue::Glue() {
	if (VBoxCGlueInit()) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Glue::~Glue() {
	VBoxCGlueTerm();
}

}
