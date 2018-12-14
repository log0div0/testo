
#include <vbox/snapshot.hpp>
#include <vbox/string.hpp>
#include <vbox/throw_if_failed.hpp>
#include <vbox/safe_array.hpp>

namespace vbox {

Snapshot::Snapshot(ISnapshot* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Snapshot::~Snapshot() {
	if (handle) {
		ISnapshot_Release(handle);
	}
}

Snapshot::Snapshot(Snapshot&& other): handle(other.handle) {
	other.handle = nullptr;
}

Snapshot& Snapshot::operator=(Snapshot&& other) {
	std::swap(handle, other.handle);

	return *this;
}

std::string Snapshot::name() const {
	try {
		BSTR result = nullptr;

		throw_if_failed(ISnapshot_GetName(handle, &result));
		return StringOut(result);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::string Snapshot::id() const {
	try {
		BSTR result = nullptr;

		throw_if_failed(ISnapshot_GetId(handle, &result));
		return StringOut(result);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::string Snapshot::getDescription() const {
	try {
		BSTR result = nullptr;

		throw_if_failed(ISnapshot_GetDescription(handle, &result));
		return StringOut(result);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void Snapshot::setDescription(const std::string& description) const {
	try {
		throw_if_failed(ISnapshot_SetDescription(handle, StringIn(description)));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::vector<Snapshot> Snapshot::children() const {
	try {
		SafeArray safe_array;
		throw_if_failed(ISnapshot_GetChildren(handle,
			SAFEARRAY_AS_OUT_PARAM(ISnapshot*, safe_array)));
		ArrayOutIface array_out = safe_array.copy_out_iface();
		return {(ISnapshot**)array_out.ifaces, (ISnapshot**)(array_out.ifaces + array_out.ifaces_count)};
	} catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}