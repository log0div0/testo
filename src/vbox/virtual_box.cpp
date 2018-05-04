
#include "virtual_box.hpp"
#include "safe_array.hpp"
#include <stdexcept>

namespace vbox {

VirtualBox::VirtualBox(IVirtualBox* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

VirtualBox::~VirtualBox() {
	if (handle) {
		IVirtualBox_Release(handle);
	}
}

VirtualBox::VirtualBox(VirtualBox&& other): handle(other.handle) {
	other.handle = nullptr;
}

VirtualBox& VirtualBox::operator=(VirtualBox&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::vector<Machine> VirtualBox::machines() const {
	try {
		SafeArray safe_array;
		HRESULT rc = IVirtualBox_get_Machines(handle, ComSafeArrayAsOutIfaceParam(safe_array.handle, IMachine*));
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return safe_array.copy_out_iface<IMachine*, Machine>();
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<std::string> VirtualBox::machine_groups() const {
	try {
		SafeArray safe_array;
		HRESULT rc = IVirtualBox_get_MachineGroups(handle, ComSafeArrayAsOutTypeParam(safe_array.handle, BSTR));
		if (FAILED(rc)) {
			throw Error(rc);
		}
		auto machine_groups = safe_array.copy_out<BSTR, StringOut>(VT_BSTR);
		std::vector<std::string> result;
		for (auto& machine_group: machine_groups) {
			result.push_back(machine_group);
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::string VirtualBox::compose_machine_filename(
	const std::string& name,
	const std::string& group,
	const std::string& create_flags,
	const std::string& base_folder
) {
	try {
		BSTR result = nullptr;
		HRESULT rc = IVirtualBox_ComposeMachineFilename(handle,
			StringIn(name),
			StringIn(group),
			StringIn(create_flags),
			StringIn(base_folder),
			&result);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return StringOut(result);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Machine VirtualBox::create_machine(
	const std::string& settings_file,
	const std::string& name,
	const std::vector<std::string>& groups_,
	const std::string& os_type_id,
	const std::string& flags
) {
	try {
		std::vector<StringIn> groups;
		for (auto& group: groups_) {
			groups.push_back(group);
		}

		SafeArray safe_array(VT_BSTR, groups.size());
		safe_array.copy_in(groups);

		IMachine* result = nullptr;
		HRESULT rc = IVirtualBox_CreateMachine(handle,
			StringIn(settings_file),
			StringIn(name),
			ComSafeArrayAsInParam(safe_array.handle, BSTR),
			StringIn(os_type_id),
			StringIn(flags),
			&result);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
