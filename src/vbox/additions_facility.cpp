
#include <vbox/additions_facility.hpp>
#include <vbox/string.hpp>
#include <vbox/throw_if_failed.hpp>

namespace vbox {

AdditionsFacility::AdditionsFacility(IAdditionsFacility* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

AdditionsFacility::~AdditionsFacility() {
	if (handle) {
		IAdditionsFacility_Release(handle);
	}
}

AdditionsFacility::AdditionsFacility(AdditionsFacility&& other): handle(other.handle) {
	other.handle = nullptr;
}

AdditionsFacility& AdditionsFacility::operator=(AdditionsFacility&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string AdditionsFacility::name() const {
	try {
		BSTR name = nullptr;
		throw_if_failed(IAdditionsFacility_get_Name(handle, &name));
		return StringOut(name);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

AdditionsFacilityType AdditionsFacility::type() const {
	try {
		AdditionsFacilityType_T result = AdditionsFacilityType_None;
		throw_if_failed(IAdditionsFacility_get_Type(handle, &result));
		return (AdditionsFacilityType)result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}


}