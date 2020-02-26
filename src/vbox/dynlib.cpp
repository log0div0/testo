
#include "dynlib.hpp"
#include <stdexcept>
#include <locale>
#include <codecvt>

namespace vbox {

DynLib::DynLib(const std::string& path) {
#ifndef WIN32
	_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#else
	using convert_type = std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convert_type, wchar_t> converter;
	_handle = LoadLibraryEx(converter.from_bytes(path).c_str(), NULL, 0);
#endif
	if (!_handle) {
		throw std::runtime_error("Failed to load dynlib " + path);
	}
}

DynLib::~DynLib() {
	if (_handle) {
#ifndef WIN32
		dlclose(_handle);
#else
		FreeLibrary(_handle);
#endif
	}
}

DynLib::DynLib(DynLib&& other): _handle(other._handle) {
	other._handle = nullptr;
}

DynLib& DynLib::operator=(DynLib&& other) {
	std::swap(_handle, other._handle);
	return *this;
}

void* DynLib::sym(const std::string& name) const {
#ifndef WIN32
    void* ptr = dlsym(_handle, name.c_str());
#else
    void* ptr = GetProcAddress(_handle, name.c_str());
#endif
    if (!ptr) {
    	throw std::runtime_error("Failed to load symbol " + name);
    }
    return ptr;
}

}
