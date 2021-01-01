
#pragma once

#include <Windows.h>
#include <string>
#include <map>

namespace winapi {

std::string get_module_file_name();
std::map<std::string, std::string> get_environment_strings();
std::wstring expand_environment_strings(const std::wstring& str);
std::string expand_environment_strings(const std::string& str);

}
