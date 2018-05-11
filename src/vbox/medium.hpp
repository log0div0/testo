
#pragma once

#include "progress.hpp"
#include <set>

namespace vbox {

struct Medium {
	Medium() = default;
	Medium(IMedium* handle);
	~Medium();

	Medium(const Medium&) = delete;
	Medium& operator=(const Medium&) = delete;
	Medium(Medium&& other);
	Medium& operator=(Medium&& other);

	std::string name() const;
	MediumState state() const;
	MediumState refresh_state() const;
	MediumVariant variant() const;
	Progress create_base_storage(size_t size, MediumVariant variant);

	operator bool() const;

	IMedium* handle = nullptr;
};

std::ostream& operator<<(std::ostream& stream, const Medium& medium);

}
