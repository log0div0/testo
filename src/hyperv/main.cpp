
#include <iostream>
#include <chrono>
#include "Connect.hpp"

void backtrace(std::ostream& stream, const std::exception& error, size_t n) {
	stream << n << ". " << error.what();
	try {
		std::rethrow_if_nested(error);
	} catch (const std::exception& error) {
		stream << std::endl;
		backtrace(stream, error, n + 1);
	} catch(...) {
		stream << std::endl;
		stream << n << ". " << "[Unknown exception type]";
	}
}

std::ostream& operator<<(std::ostream& stream, const std::exception& error) {
	backtrace(stream, error, 1);
	return stream;
}

void main() {
	try {
		wmi::CoInitializer initializer;
		initializer.initalize_security();
		hyperv::Connect connect;

		auto start = std::chrono::high_resolution_clock::now();
		std::cout << connect.machine("ubuntu").display().width() << std::endl;
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time = end - start;
		std::cout << time.count() << " seconds" << std::endl;

	} catch (const std::exception& error) {
		std::cerr << error << std::endl;
	}
}
