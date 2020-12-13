
#pragma once

#include <Windows.h>
#include <string>
#include <stdexcept>
#include <sstream>

namespace wmi {

template <typename T>
std::string to_hex(T t) {
	std::stringstream stream;
	stream << std::hex << "0x" << t;
	return stream.str();
}

struct ComError: std::runtime_error {
	ComError(HRESULT hresult_):
		std::runtime_error("HRESULT = " + to_hex(hresult_)),
		hresult(hresult_)
	{

	}

	HRESULT hresult;
};

inline void throw_if_failed(HRESULT hr) {
	if (FAILED(hr)) {
		throw ComError(hr);
	}
}

}
