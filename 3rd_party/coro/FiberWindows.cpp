
#include "coro/FiberWindows.h"
#include <cassert>
#include <system_error>
thread_local void* t_fiber = nullptr;

#if _WIN32_WINNT >= _WIN32_WINNT_WIN7

#ifdef _DEBUG
#define CORO_STACK_COMMIT_SIZE 1024 * 1024 * 10
#define CORO_STACK_RESERVE_SIZE 1024 * 1024 * 40
#else
#define CORO_STACK_COMMIT_SIZE 1024 * 1024 * 1
#define CORO_STACK_RESERVE_SIZE 1024 * 1024 * 4
#endif

#else

#ifdef _DEBUG
#define CORO_STACK_SIZE 1024 * 1024 * 40
#else
#define CORO_STACK_SIZE 1024 * 1024 * 4
#endif

#endif

namespace coro {

void Fiber::initialize() {
	assert(!t_fiber);
#if _WIN32_WINNT >= _WIN32_WINNT_WIN7
	t_fiber = ConvertThreadToFiberEx(0, FIBER_FLAG_FLOAT_SWITCH);
#else
	t_fiber = ConvertThreadToFiber(0);
#endif
}

void Fiber::deinitialize() {
	assert(t_fiber);
	ConvertFiberToThread();
}

Fiber::Fiber(LPFIBER_START_ROUTINE startRoutine, LPVOID parameter) {
#if _WIN32_WINNT >= _WIN32_WINNT_WIN7
    _fiber = CreateFiberEx(CORO_STACK_COMMIT_SIZE, CORO_STACK_RESERVE_SIZE, FIBER_FLAG_FLOAT_SWITCH, startRoutine, parameter);
#else
    _fiber = CreateFiber(CORO_STACK_SIZE, startRoutine, parameter);
#endif
	if (_fiber == 0) {
		throw std::system_error(GetLastError(), std::system_category());
	}
}

Fiber::~Fiber() {
	DeleteFiber(_fiber);
}

void Fiber::enter() {
	SwitchToFiber(_fiber);
}

void Fiber::switchTo(Fiber& fiber) {
	SwitchToFiber(fiber._fiber);
}

void Fiber::exit() {
	assert(t_fiber);
	SwitchToFiber(t_fiber);
}

}