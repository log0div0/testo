
#include "medium_attachment.hpp"
#include <stdexcept>
#include <ostream>
#include "error.hpp"
#include "string.hpp"

namespace vbox {

MediumAttachment::MediumAttachment(IMediumAttachment* handle): handle(handle) {
}

MediumAttachment::~MediumAttachment() {
	if (handle) {
		IMediumAttachment_Release(handle);
	}
}

MediumAttachment::MediumAttachment(MediumAttachment&& other): handle(other.handle) {
	other.handle = nullptr;
}

MediumAttachment& MediumAttachment::operator=(MediumAttachment&& other) {
	std::swap(handle, other.handle);
	return *this;
}

Medium MediumAttachment::medium() const {
	try {
		IMedium* result = nullptr;
		HRESULT rc = IMediumAttachment_get_Medium(handle, &result);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		if (result) {
			return result;
		} else {
			return {};
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

std::string MediumAttachment::controller() const {
	try {
		BSTR result = nullptr;
		HRESULT rc = IMediumAttachment_get_Controller(handle, &result);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return StringOut(result);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

LONG MediumAttachment::port() const {
	try {
		LONG result = 0;
		HRESULT rc = IMediumAttachment_get_Port(handle, &result);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

LONG MediumAttachment::device() const {
	try {
		LONG result = 0;
		HRESULT rc = IMediumAttachment_get_Device(handle, &result);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

DeviceType MediumAttachment::type() const {
	try {
		DeviceType result = DeviceType_Null;
#ifdef WIN32
		HRESULT rc = IMediumAttachment_get_Type(handle, &result);
#else
		HRESULT rc = IMediumAttachment_get_Type(handle, (uint32_t*)&result);
#endif
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}

}

std::ostream& operator<<(std::ostream& stream, const MediumAttachment& medium_attachment) {
	Medium medium = medium_attachment.medium();
	stream << "medium={";
	if (medium) {
		stream << medium;
	} else {
		stream << "NONE";
	}
	stream << "}";
	return stream
		<< " controller=" << medium_attachment.controller()
		<< " port=" << medium_attachment.port()
		<< " device=" << medium_attachment.device()
		<< " type=" << medium_attachment.type();
}

}
