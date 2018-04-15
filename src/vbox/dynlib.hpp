
#pragma once

#ifdef WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif
#include <string>

namespace vbox {

struct DynLib {
	DynLib() = default;
	DynLib(const std::string& path);
	~DynLib();

	DynLib(const DynLib&) = delete;
	DynLib& operator=(const DynLib&) = delete;

	DynLib(DynLib&& other);
	DynLib& operator=(DynLib&& other);

	void* sym(const std::string& name) const;

#ifndef WIN32
	using Handle = void*;
#else
	using Handle = HMODULE;
#endif

private:
	Handle _handle = nullptr;
};

}
