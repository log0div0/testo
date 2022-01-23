
#include "Exceptions.hpp"

std::string GetFailureCategory(const std::exception& error) {
	// exception with category should be on the bottom of the stack
	try {
		std::rethrow_if_nested(error);
		const ExceptionWithCategory* exception = dynamic_cast<const ExceptionWithCategory*>(&error);
		if (exception) {
			return exception->failure_category;
		} else {
			return {};
		}
	} catch (const std::exception& error) {
		return GetFailureCategory(error);
	}
}
