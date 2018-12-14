
#include <vbox/safe_array.hpp>
#include <vbox/guest.hpp>
#include <vbox/throw_if_failed.hpp>
#include <vbox/string.hpp>

namespace vbox {

Guest::Guest(IGuest* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Guest::~Guest() {
	if (handle) {
		IGuest_Release(handle);
	}
}

Guest::Guest(Guest&& other): handle(other.handle) {
	other.handle = nullptr;
}

Guest& Guest::operator=(Guest&& other) {
	std::swap(handle, other.handle);
	return *this;
}

GuestSession Guest::create_session(const std::string& user,
		const std::string& password,
		const std::string& domain,
		const std::string& name)
{
	try {
		IGuestSession* result = nullptr;
		throw_if_failed(IGuest_CreateSession(handle,
			StringIn(user),
			StringIn(password),
			StringIn(domain),
			StringIn(name),
			&result));

		GuestSession gs(result);
		if (gs.wait_for(GuestSessionWaitForFlag_Start, 5000) == GuestSessionWaitResult_Start) {
			return gs;
		} else {
			throw std::runtime_error("Guest session wait for start error");
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<AdditionsFacility> Guest::facilities() const {
	try {
		SafeArray safe_array;
		throw_if_failed(IGuest_GetFacilities(handle, SAFEARRAY_AS_OUT_PARAM(IAdditionsFacility*, safe_array)));
		ArrayOutIface array_out = safe_array.copy_out_iface();
		return {(IAdditionsFacility**)array_out.ifaces, (IAdditionsFacility**)(array_out.ifaces + array_out.ifaces_count)};
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}


}