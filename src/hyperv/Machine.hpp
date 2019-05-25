
#pragma once

#include "Display.hpp"
#include "StorageController.hpp"
#include "Keyboard.hpp"
#include "NIC.hpp"

namespace hyperv {

struct Machine {
	enum class State: uint16_t {
		Unknown = 0,
		Other = 1,
		Enabled = 2,
		Disabled = 3,
		Paused = 32768
	};

	Machine(wmi::WbemClassObject computerSystem_, wmi::WbemServices services_);

	std::string name() const;
	State state() const;
	Display display() const;
	void destroy();

	void setNotes(const std::vector<std::string>& notes);
	std::vector<std::string> notes() const;

	void requestStateChange(State requestedState);
	void start();
	void stop();
	void pause();

	std::vector<StorageController> ideControllers() const;
	std::vector<StorageController> controllers(const std::string& subtype) const;
	Keyboard keyboard() const;

	NIC addNIC(const std::string& name, bool legacy = true);

	wmi::WbemClassObject computerSystem;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;
};

}
