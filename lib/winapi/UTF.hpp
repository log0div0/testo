
#pragma once

#include <Windows.h>
#include <string>

namespace winapi {

std::wstring utf8_to_utf16(const std::string& utf8);
std::string utf16_to_utf8(const std::wstring& utf16);

}
