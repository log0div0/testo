
#pragma once

#include <ucontext.h>
#include <cstdint>
#include <vector>

namespace coro {

class Fiber {
public:
	static void initialize();
	static void deinitialize();

	Fiber(void (*startRoutine)(void*), void* parameter);
	~Fiber();

	Fiber(const Fiber& other) = delete;
	Fiber(Fiber&& other);

	Fiber& operator=(const Fiber& other) = delete;
	Fiber& operator=(Fiber&& other);

	void enter();
	void switchTo(Fiber& fiber);
	void exit();

private:
	ucontext_t _context;
	std::vector<uint8_t> _buffer;
};

}