
#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include <memory>
#include <ostream>

struct Exception: std::exception {
	Exception(std::string what): msg(std::move(what)) {}

	const char* what() const noexcept override {
		return msg.c_str();
	}

protected:
	std::string msg;
};

struct ExceptionWithPos: Exception {
	template <typename Pos>
	ExceptionWithPos(const Pos& pos, const std::string original_msg_):
		Exception(std::string(pos) + ": " + original_msg_),
		original_msg(std::move(original_msg_))
	{
	}

	std::string original_msg;
};

struct TestFailedException: Exception {
	TestFailedException(): Exception("At least one of the tests failed") {}
};

struct ContinueError: std::runtime_error {
	using std::runtime_error::runtime_error;
};

std::ostream& operator<<(std::ostream& stream, const std::exception& error);
