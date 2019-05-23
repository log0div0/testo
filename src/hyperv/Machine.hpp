
#pragma once

#include "Display.hpp"

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

	wmi::WbemClassObject computerSystem;
	wmi::WbemServices services;

private:
	wmi::WbemClassObject settings() const;
};

}
