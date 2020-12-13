
#include "Functions.hpp"
#include "UTF.hpp"
#include <stdexcept>

namespace winapi {

std::string get_module_file_name() {
	TCHAR szFileName[MAX_PATH] = {};
	GetModuleFileName(NULL, szFileName, MAX_PATH);
	return utf16_to_utf8(szFileName);
}

}