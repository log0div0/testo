
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

std::ostream& operator<<(std::ostream& stream, const MediumAttachment& medium_attachment) {
	Medium medium = medium_attachment.medium();
	if (medium) {
		stream << medium;
	} else {
		stream << "EMPTY";
	}
	return stream;
}

}
