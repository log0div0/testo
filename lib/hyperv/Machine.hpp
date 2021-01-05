
#pragma once

#include "Display.hpp"
#include "StorageController.hpp"
#include "Keyboard.hpp"
#include "NIC.hpp"
#include "Snapshot.hpp"
#include "Processor.hpp"
#include "Memory.hpp"

namespace hyperv {

struct Machine {
	enum class State: uint16_t {
		Unknown = 0,
		Other = 1,
		Enabled = 2,
		Disabled = 3,
		Offline = 6,
		Quiesce = 9
	};

	Machine(wmi::WbemClassObject computerSystem_, wmi::WbemServices services_);

	std::string name() const;
	State state() const;
	Display display() const;
	void destroy();

	void setNotes(const std::vector<std::string>& notes);
	std::vector<std::string> notes() const;

	void requestStateChange(State requestedState);
	void enable();
	void disable();

	std::vector<StorageController> ideControllers() const;
	std::vector<StorageController> controllers(const std::string& subtype) const;
	Keyboard keyboard() const;
	Processor processor() const;
	Memory memory() const;

	NIC addNIC(const std::string& name, bool legacy = true);

	Snapshot createSnapshot();
	std::vector<Snapshot> snapshots();
	Snapshot snapshot(const std::string& name);

	wmi::WbemClassObject computerSystem;
	wmi::WbemClassObject virtualSystemSettingData;
	wmi::WbemServices services;

private:
	wmi::WbemClassObject activeSettings();
};

}
