
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
		return safe_array.copy_out_iface_param<IMachine*, Machine>();
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<String> VirtualBox::machine_groups() const {
	try {
		SafeArray safe_array;
		HRESULT rc = IVirtualBox_get_MachineGroups(handle, ComSafeArrayAsOutTypeParam(safe_array.handle, BSTR));
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return safe_array.copy_out_param<BSTR, String>(VT_BSTR);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

String VirtualBox::compose_machine_filename(
	const Utf16String& name,
	const Utf16String& group,
	const Utf16String& create_flags,
	const Utf16String& base_folder
) {
	try {
		String result;
		HRESULT rc = IVirtualBox_ComposeMachineFilename(handle, name.data, group.data, create_flags.data, base_folder.data, &result.data);
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
