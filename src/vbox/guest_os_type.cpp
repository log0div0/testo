
#include <vbox/guest_os_type.hpp>
#include <vbox/throw_if_failed.hpp>
#include <vbox/string.hpp>

namespace vbox {

GuestOSType::GuestOSType(IGuestOSType* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

GuestOSType::~GuestOSType() {
	if (handle) {
		IGuestOSType_Release(handle);
	}
}

GuestOSType::GuestOSType(GuestOSType&& other): handle(other.handle) {
	other.handle = nullptr;
}

GuestOSType& GuestOSType::operator=(GuestOSType&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string GuestOSType::id() const {
	try {
		BSTR result = nullptr;
		throw_if_failed(IGuestOSType_get_Id(handle, &result));
		return StringOut(result);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::string GuestOSType::description() const {
	try {
		BSTR result = nullptr;
		throw_if_failed(IGuestOSType_get_Description(handle, &result));
		return StringOut(result);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::string GuestOSType::family_id() const {
	try {
		BSTR result = nullptr;
		throw_if_failed(IGuestOSType_get_FamilyId(handle, &result));
		return StringOut(result);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::string GuestOSType::family_description() const {
	try {
		BSTR result = nullptr;
		throw_if_failed(IGuestOSType_get_FamilyDescription(handle, &result));
		return StringOut(result);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

ULONG GuestOSType::recommended_ram() const {
	try {
		ULONG result = 0;
		throw_if_failed(IGuestOSType_get_RecommendedRAM(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

ULONG GuestOSType::recommended_vram() const {
	try {
		ULONG result = 0;
		throw_if_failed(IGuestOSType_get_RecommendedVRAM(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::ostream& operator<<(std::ostream& stream, const GuestOSType& guest_os_type) {
	stream
		<< "id=" << guest_os_type.id()
		<< " description=" << guest_os_type.description()
		<< " family_id=" << guest_os_type.family_id()
		<< " family_description=" << guest_os_type.family_description()
		<< " recommended_ram=" << guest_os_type.recommended_ram() << "Mb"
		<< " recommended_vram=" << guest_os_type.recommended_vram() << "Mb"
	;
	return stream;
}

}
