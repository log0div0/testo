
#pragma once

#include "Display.hpp"
#include "StorageController.hpp"
#include "Keyboard.hpp"

namespace hyperv {

struct Machine {
	Machine(wmi::WbemClassObject computerSystem_, wmi::WbemServices services_);

	std::string name() const;
	bool is_running() const;
	Display display() const;
	void destroy();

	void setNotes(const std::vector<std::string>& notes);
	std::vector<std::string> notes() const;

	void requestStateChange(uint16_t requestedState);
	void start();
	void stop();
	void pause();

	std::vector<StorageController> ideControllers() const;
	std::vector<StorageController> controllers(const std::string& subtype) const;
	Keyboard keyboard() const;

	wmi::WbemClassObject computerSystem;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
