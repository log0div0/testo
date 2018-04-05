
#include "error.hpp"
#include <sstream>

namespace vbox {

Error::Error(HRESULT rc) {
	std::stringstream ss;
	ss << "Error code: " << std::hex << rc << std::dec;
	_what = ss.str();
}

Error::~Error() {}

const char* Error::what() const noexcept {
	return _what.c_str();
}

}
