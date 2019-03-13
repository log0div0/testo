
#pragma once

#include <Windows.h>

namespace coro {

/// Wrapper вокруг Windows Fibers
class Fiber {
public:
	/// ConvertThreadToFiber
	static void initialize();
	/// ConvertFiberToThread
	static void deinitialize();

	/// CreateFiber
	Fiber(LPFIBER_START_ROUTINE startRoutine, LPVOID parameter);
	/// DeleteFiber
	~Fiber();

	Fiber(const Fiber& other) = delete;
	Fiber(Fiber&& other);

	Fiber& operator=(const Fiber& other) = delete;
	Fiber& operator=(Fiber&& other);

	void enter();
	void switchTo(Fiber& fiber);
	void exit();

private:
	void* _fiber;
};

}