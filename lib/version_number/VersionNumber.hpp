
#pragma once

#include <string>

struct VersionNumber {
	int MAJOR = 0;
	int MINOR = 0;
	int PATCH = 0;

	VersionNumber() = default;
	VersionNumber(int a, int b, int c):
		MAJOR(a),
		MINOR(b),
		PATCH(c) {}
	VersionNumber(const std::string& str);

	bool operator<(const VersionNumber& other);

	std::string to_string() const;
};
