
#include "Exceptions.hpp"
#include <system_error>

#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;

#ifdef WIN32
#include <winapi/Functions.hpp>
#endif

void backtrace(std::ostream& stream, const std::exception& error) {
#ifdef WIN32
	if (dynamic_cast<const std::system_error*>(&error) && !dynamic_cast<const fs::filesystem_error*>(&error)) {
		std::wstring utf16_err = winapi::acp_to_utf16(error.what());
		std::string utf8_err = winapi::utf16_to_utf8(utf16_err);
		stream << utf8_err;
	} else {
		stream << error.what();
	}
#else
	stream << error.what();
#endif
	try {
		std::rethrow_if_nested(error);
	} catch (const Exception& error) {
		stream << "\n";
		backtrace(stream, error);
	} catch (const std::exception& error) {
		stream << "\n\t- ";
		backtrace(stream, error);
	} catch(...) {
		stream << std::endl;
		stream << "[Unknown exception type]";
	}
}

std::ostream& operator<<(std::ostream& stream, const std::exception& error) {
	backtrace(stream, error);
	return stream;
}
