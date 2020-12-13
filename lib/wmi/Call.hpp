
#pragma once

#include "WbemServices.hpp"
#include "Variant.hpp"
#include "WbemClassObject.hpp"

namespace wmi {

struct Call {
	Call(WbemServices services_, std::string class_name_, std::string method_name_);

	Call& with(std::string name, const Variant& value);
	Call& with(std::string name, const WbemClassObject& object);
	Call& with(std::string name, const std::vector<WbemClassObject>& objects);

	WbemClassObject exec(WbemClassObject object);
	WbemClassObject exec();

private:
	WbemServices services;
	std::string class_name;
	std::string method_name;
	WbemClassObject method_instance;
};

}