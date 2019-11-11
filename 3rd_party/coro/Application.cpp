
#include "coro/Application.h"
#ifdef _MSC_VER
#include <ObjBase.h>
#endif

namespace coro {

Application::Application(const std::function<void()>& main):
	_coro(main)
{
	_ioService.post([=] {
		_coro.start();
	});
}

Application::~Application() {
}

void Application::run() {
	_ioService.run();
}

}
