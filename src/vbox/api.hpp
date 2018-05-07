
#pragma once

#include "enums.hpp"
#include "dynlib.hpp"

#ifdef unix
#define IF_UNIX(X) X
#else
#define IF_UNIX(X)
#endif

#ifdef WIN32
#define IF_WIN32(X) X
#else
#define IF_WIN32(X)
#endif

namespace vbox {

struct API {
	API();
	~API();

	API(const API&) = delete;
	API& operator=(const API&) = delete;

private:
	DynLib _dynlib;
};

extern PCVBOXCAPI api;

}
