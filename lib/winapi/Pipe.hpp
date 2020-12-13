
#pragma once

#include <Windows.h>
#include <utility>

namespace winapi {

struct Pipe {
	static std::pair<Pipe, Pipe> create();

	Pipe() = default;
	Pipe(HANDLE handle_);
	~Pipe();

	Pipe(Pipe&& other);
	Pipe& operator=(Pipe&& other);

	void setHandleInformation(DWORD dwMask, DWORD dwFlags);

	HANDLE handle = NULL;
};

}
