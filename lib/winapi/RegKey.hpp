
#pragma once

#include <Windows.h>
#include <string>

namespace winapi {

struct RegKey {
	RegKey(HKEY key, const std::string& path);
	~RegKey();

	RegKey(RegKey&&);
	RegKey& operator=(RegKey&&);

	std::string query_str(const std::string& name) const;
	void set_expand_str(const std::string& name, const std::string& value);

private:
	HKEY handle = NULL;
};

}
