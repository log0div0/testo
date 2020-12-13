
#pragma once

#include <string>
#include "WbemClassObject.hpp"
#include <wbemcli.h>

namespace wmi {

struct WbemObjectTextSrc: Object<IWbemObjectTextSrc> {
	WbemObjectTextSrc();
	std::string getText(const WbemClassObject& object, ULONG format = WMI_OBJ_TEXT_WMI_DTD_2_0);
};

}
