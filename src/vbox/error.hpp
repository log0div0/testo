
#pragma once

#include "api.hpp"
#include <stdexcept>
#include <string>

namespace vbox {

struct Error: public std::exception {
	Error(HRESULT rc);
	virtual ~Error();

	virtual const char* what() const noexcept;

private:
	std::string _what;
};

}
