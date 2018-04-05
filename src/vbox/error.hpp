
#pragma once

#include <VBoxCAPIGlue/VBoxCAPIGlue.h>
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
