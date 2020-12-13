
#include "Pipe.hpp"
#include <stdexcept>

namespace winapi {

std::pair<Pipe, Pipe> Pipe::create() {
	SECURITY_ATTRIBUTES saAttr = {};
	saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
	saAttr.bInheritHandle = TRUE;
	saAttr.lpSecurityDescriptor = NULL;

	HANDLE read = NULL;
	HANDLE write = NULL;

	if (!CreatePipe(&read, &write, &saAttr, 0)) {
		throw std::runtime_error("CreatePipe failed");
	}

	return {read, write};
}

Pipe::Pipe(HANDLE handle_): handle(handle_) {

}

Pipe::~Pipe() {
	if (handle) {
		CloseHandle(handle);
		handle = NULL;
	}
}


Pipe::Pipe(Pipe&& other): handle(other.handle) {
	other.handle = NULL;
}

Pipe& Pipe::operator=(Pipe&& other) {
	std::swap(handle, other.handle);
	return *this;
}


void Pipe::setHandleInformation(DWORD dwMask, DWORD dwFlags) {
	if (!SetHandleInformation(handle, dwMask, dwFlags)) {
		throw std::runtime_error("SetHandleInformation failed");
	}
}

}
