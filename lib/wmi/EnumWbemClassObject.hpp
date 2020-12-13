
#pragma once

#include "Object.hpp"
#include "WbemClassObject.hpp"
#include <wbemcli.h>

namespace wmi {

struct EnumWbemClassObject: Object<IEnumWbemClassObject> {
	using Object<IEnumWbemClassObject>::Object;

	std::vector<WbemClassObject> next(size_t count);
	std::vector<WbemClassObject> getAll(size_t batch_size = 1);
	WbemClassObject getOne();
};

}
