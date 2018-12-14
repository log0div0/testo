
#include <vbox/medium_attachment.hpp>
#include <stdexcept>
#include <ostream>
#include <vbox/throw_if_failed.hpp>
#include <vbox/string.hpp>

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
		throw_if_failed(IMediumAttachment_get_Medium(handle, &result));
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
		throw_if_failed(IMediumAttachment_get_Controller(handle, &result));
		return StringOut(result);
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

LONG MediumAttachment::port() const {
	try {
		LONG result = 0;
		throw_if_failed(IMediumAttachment_get_Port(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

LONG MediumAttachment::device() const {
	try {
		LONG result = 0;
		throw_if_failed(IMediumAttachment_get_Device(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

DeviceType MediumAttachment::type() const {
	try {
		DeviceType_T result = DeviceType_Null;
		throw_if_failed(IMediumAttachment_get_Type(handle, &result));
		return (DeviceType)result;
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
