
#pragma once

#include "Object.hpp"
#include "WbemServices.hpp"
#include <string>
#include <wbemcli.h>

namespace wmi {

struct WbemLocator: Object<IWbemLocator> {
	WbemLocator();
	WbemServices connectServer(const std::string& path);
};

}
