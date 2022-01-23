
#pragma once

#include <stdexcept>

struct ExceptionWithCategory: std::runtime_error {
	ExceptionWithCategory(const std::string& message, const std::string& failure_category_):
		std::runtime_error(message), failure_category(failure_category_) {}

	std::string failure_category;
};

struct LogicError: ExceptionWithCategory {
	LogicError(const std::string& message): ExceptionWithCategory(message, "logic_error") {}
};

struct ContinueError: std::runtime_error {
	using std::runtime_error::runtime_error;
};

std::string GetFailureCategory(const std::exception& error);