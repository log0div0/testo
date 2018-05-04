
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
		ArrayOut array_out;
		rc = api->pfnSafeArrayCopyOutIfaceParamHelper((IUnknown***)&array_out.values, &array_out.values_count, safe_array.handle);
		if (FAILED(rc)) {
			throw Error(rc);
		}
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

std::vector<std::string> VirtualBox::machine_groups() const {
	try {
		SafeArray safe_array;
		HRESULT rc = IVirtualBox_get_MachineGroups(handle, ComSafeArrayAsOutTypeParam(safe_array.handle, BSTR));
		if (FAILED(rc)) {
			throw Error(rc);
		}
		ArrayOut array_out;
		rc = api->pfnSafeArrayCopyOutParamHelper((void**)&array_out.values, &array_out.values_count, VT_BSTR, safe_array.handle);
		if (FAILED(rc)) {
			throw Error(rc);
		}
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

		SafeArray safe_array(VT_BSTR, bstrs.size());

		HRESULT rc = api->pfnSafeArrayCopyInParamHelper(safe_array.handle, bstrs.data(), bstrs.size() * sizeof(BSTR));
		if (FAILED(rc)) {
			throw Error(rc);
		}

		IMachine* result = nullptr;
		rc = IVirtualBox_CreateMachine(handle,
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
