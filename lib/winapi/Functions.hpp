
#pragma once

#include <Windows.h>
#include <string>
#include <map>

namespace winapi {

void throw_error(const std::string& ascii_desc, DWORD error_code);

std::wstring multi_byte_to_wide_char(const std::string& multi_byte, UINT codepage);
std::string wide_char_to_multi_byte(const std::wstring& wide_char, UINT codepage);
std::wstring utf8_to_utf16(const std::string& utf8);
std::wstring acp_to_utf16(const std::string& utf8);
std::string utf16_to_utf8(const std::wstring& utf16);

std::string get_module_file_name();
std::map<std::string, std::string> get_environment_strings();
std::wstring expand_environment_strings(const std::wstring& str);
std::string expand_environment_strings(const std::string& str);

}
