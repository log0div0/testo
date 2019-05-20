
#include "Hypervisor.hpp"
#include "Vbox.hpp"

#ifdef WIN32
#include "windows/HyperV.hpp"
#endif

#ifdef __linux__
#include "linux/Qemu.hpp"
#endif

Guest::Guest(const std::string& name): _name(std::move(name)) {

}

std::shared_ptr<Hypervisor> Hypervisor::get(const std::string& name) {
	if (!name.size()) {
		#ifdef WIN32
			return std::make_shared<HyperV>();
		#elif __linux__
			return std::make_shared<Qemu>();
		#else
			return std::make_shared<Vbox>();
		#endif
	}
	if (name == "vbox") {
		return std::make_shared<Vbox>();
	}

#ifdef WIN32
	if (name == "hyperv") {
		return std::make_shared<HyperV>();
	}
#endif

#ifdef __linux__
	if (name == "qemu") {
		return std::make_shared<Qemu>();
	}
#endif

	throw std::runtime_error("Unknown hypervisor '" + name + "'");
}
