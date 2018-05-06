
#pragma once

#include "medium.hpp"

namespace vbox {

struct MediumAttachment {
	MediumAttachment(IMediumAttachment* handle);
	~MediumAttachment();

	MediumAttachment(const MediumAttachment&) = delete;
	MediumAttachment& operator=(const MediumAttachment&) = delete;
	MediumAttachment(MediumAttachment&& other);
	MediumAttachment& operator=(MediumAttachment&& other);

	Medium medium() const;
	std::string controller() const;
	int port() const;
	int device() const;
	DeviceType type() const;

	IMediumAttachment* handle = nullptr;
};

std::ostream& operator<<(std::ostream& stream, const MediumAttachment& medium_attachment);

}
