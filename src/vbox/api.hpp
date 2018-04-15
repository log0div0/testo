
#pragma once

#include <VBoxCAPI/VBoxCAPI_v5_2.h>
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
