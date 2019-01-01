
#pragma once

#include "enums.hpp"
#include "dynlib.hpp"

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
