
#include "coro/FiberLinux.h"
#include <cassert>
#include <system_error>

thread_local ucontext_t* mainContext = nullptr;

namespace coro {

void Fiber::initialize() {
	assert(mainContext == nullptr);
	mainContext = new ucontext_t();
	if (getcontext(mainContext) != 0) {
		throw std::system_error(errno, std::system_category());
	};
}


void Fiber::deinitialize() {
	assert(mainContext != nullptr);
	delete mainContext;
	mainContext = nullptr;
}

Fiber::Fiber(void (*startRoutine)(void*), void* parameter): _buffer(1024 * 1024 * 4) {
	if (getcontext(&_context) != 0) {
		throw std::system_error(errno, std::system_category());
	};
	_context.uc_stack.ss_sp = _buffer.data();
	_context.uc_stack.ss_size = _buffer.size();
	_context.uc_link = nullptr;
	makecontext(&_context, (void (*)())startRoutine, 1, parameter);
}

Fiber::~Fiber() {
	// do nothing
}

void Fiber::enter() {
	if (swapcontext(mainContext, &_context) != 0) {
		throw std::system_error(errno, std::system_category());
	}
}

void Fiber::switchTo(Fiber& fiber) {
	if (swapcontext(&_context, &fiber._context) != 0) {
		throw std::system_error(errno, std::system_category());
	}
}

void Fiber::exit() {
	if (swapcontext(&_context, mainContext) != 0) {
		throw std::system_error(errno, std::system_category());
	}
}

}