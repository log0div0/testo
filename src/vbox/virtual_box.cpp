
#include "virtual_box.hpp"
#include "safe_array.hpp"
#include "throw_if_failed.hpp"
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
		throw_if_failed(IVirtualBox_get_Machines(handle, ComSafeArrayAsOutIfaceParam(safe_array.handle, IMachine*)));
		ArrayOut array_out = safe_array.copy_out();
		std::vector<Machine> result;
		for (ULONG i = 0; i < array_out.values_count; ++i) {
			result.push_back(Machine(((IMachine**)array_out.values)[i]));
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<Medium> VirtualBox::dvd_images() const {
	try {
		SafeArray safe_array;
		throw_if_failed(IVirtualBox_get_DVDImages(handle, ComSafeArrayAsOutIfaceParam(safe_array.handle, IMedium*)));
		ArrayOut array_out = safe_array.copy_out();
		std::vector<Medium> result;
		for (ULONG i = 0; i < array_out.values_count; ++i) {
			result.push_back(Medium(((IMedium**)array_out.values)[i]));
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<Medium> VirtualBox::hard_disks() const {
	try {
		SafeArray safe_array;
		throw_if_failed(IVirtualBox_get_HardDisks(handle, ComSafeArrayAsOutIfaceParam(safe_array.handle, IMedium*)));
		ArrayOut array_out = safe_array.copy_out();
		std::vector<Medium> result;
		for (ULONG i = 0; i < array_out.values_count; ++i) {
			result.push_back(Medium(((IMedium**)array_out.values)[i]));
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<GuestOSType> VirtualBox::guest_os_types() const {
	try {
		SafeArray safe_array;
		throw_if_failed(IVirtualBox_get_GuestOSTypes(handle, ComSafeArrayAsOutIfaceParam(safe_array.handle, IGuestOSType*)));
		ArrayOut array_out = safe_array.copy_out();
		std::vector<GuestOSType> result;
		for (ULONG i = 0; i < array_out.values_count; ++i) {
			result.push_back(GuestOSType(((IGuestOSType**)array_out.values)[i]));
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Machine VirtualBox::find_machine(const std::string& name) const {
	try {
		IMachine* result = nullptr;
		throw_if_failed(IVirtualBox_FindMachine(handle, StringIn(name), &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

GuestOSType VirtualBox::get_guest_os_type(const std::string& name) const {
	try {
		IGuestOSType* result = nullptr;
		throw_if_failed(IVirtualBox_GetGuestOSType(handle, StringIn(name), &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<std::string> VirtualBox::machine_groups() const {
	try {
		SafeArray safe_array;
		throw_if_failed(IVirtualBox_get_MachineGroups(handle, ComSafeArrayAsOutTypeParam(safe_array.handle, BSTR)));
		ArrayOut array_out = safe_array.copy_out(VT_BSTR);
		std::vector<std::string> result;
		for (ULONG i = 0; i < array_out.values_count / sizeof(BSTR); ++i) {
			result.push_back(StringOut(((BSTR*)array_out.values)[i]));
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
) const {
	try {
		BSTR result = nullptr;
		throw_if_failed(IVirtualBox_ComposeMachineFilename(handle,
			StringIn(name),
			StringIn(group),
			StringIn(create_flags),
			StringIn(base_folder),
			&result));
		return StringOut(result);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Machine VirtualBox::create_machine(
	const std::string& settings_file,
	const std::string& name,
	const std::vector<std::string>& groups,
	const std::string& os_type_id,
	const std::string& flags
) {
	try {
		std::vector<StringIn> strings_in;
		for (auto& group: groups) {
			strings_in.push_back(group);
		}
		std::vector<BSTR> bstrs;
		for (auto& string_in: strings_in) {
			bstrs.push_back(string_in);
		}

		SafeArray safe_array(VT_BSTR, (ULONG)bstrs.size());
		safe_array.copy_in(bstrs.data(), (ULONG)(bstrs.size() * sizeof(BSTR)));

		IMachine* result = nullptr;
		throw_if_failed(IVirtualBox_CreateMachine(handle,
			StringIn(settings_file),
			StringIn(name),
			ComSafeArrayAsInParam(safe_array.handle, BSTR),
			StringIn(os_type_id),
			StringIn(flags),
			&result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Medium VirtualBox::create_medium(
	const std::string& format,
	const std::string& location,
	AccessMode access_mode,
	DeviceType device_type
) {
	try {
		IMedium* result = nullptr;
		throw_if_failed(IVirtualBox_CreateMedium(handle,
			StringIn(format),
			StringIn(location),
			access_mode,
			device_type,
			&result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Medium VirtualBox::open_medium(
	const std::string& location,
	DeviceType device_type,
	AccessMode access_mode,
	bool force_new_uuid
) {
	try {
		IMedium* result = nullptr;
		throw_if_failed(IVirtualBox_OpenMedium(handle,
			StringIn(location),
			device_type,
			access_mode,
			force_new_uuid,
			&result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void VirtualBox::register_machine(const Machine& machine) {
	try {
		throw_if_failed(IVirtualBox_RegisterMachine(handle, machine.handle));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
