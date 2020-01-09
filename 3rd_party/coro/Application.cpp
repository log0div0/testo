
#include "coro/Application.h"
#include "coro/Work.h"

namespace coro {

Application::Application(const std::function<void()>& main):
	_coro([=] {
		Work work;
		main();
	})
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
