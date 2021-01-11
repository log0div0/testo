
#pragma once

#include <Windows.h>
#include <string>
#include <vector>

namespace winapi {

struct RegKey {
	RegKey(HKEY key, const std::string& path, REGSAM sam_desired = KEY_ALL_ACCESS);
	~RegKey();

	RegKey(RegKey&&);
	RegKey& operator=(RegKey&&);

	void set_expand_str(const std::string& name, const std::string& value);
	std::vector<std::string> enum_values() const;
	std::string get_str(const std::string& name) const;

private:
	HKEY handle = NULL;
};

}
