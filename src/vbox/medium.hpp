
#pragma once

#include "api.hpp"
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
	std::set<MediumVariant> variant() const;

	operator bool() const;

	IMedium* handle = nullptr;
};

std::ostream& operator<<(std::ostream& stream, const Medium& medium);

}
