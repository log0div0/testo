
#pragma once

#include "api.hpp"
#include <string>

namespace vbox {

struct StringIn {
	StringIn(const std::string& str);
	~StringIn();

	operator BSTR() const;

	StringIn(const StringIn&) = delete;
	StringIn& operator=(const StringIn&) = delete;
	StringIn(StringIn&&);
	StringIn& operator=(StringIn&&);

private:
	BSTR data = nullptr;
};

struct StringOut {
	StringOut(BSTR data);
	~StringOut();

	operator std::string() const;

	StringOut(const StringOut&) = delete;
	StringOut& operator=(const StringOut&) = delete;
	StringOut(StringOut&&);
	StringOut& operator=(StringOut&&);

private:
	char* data = nullptr;
};

}
