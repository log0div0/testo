
#pragma once

#include <functional>
#include <list>

namespace coro {

/*!
	@brief Выполняет действия при выходе из области видимости

	Пример 1:
	@code
		void main() {
			setupLogging();
			Finally cleanup([] {
				teardownLogging();
			});
			// do some stuff
		}
	@endcode

	Пример 2:
	@code
		Finally rollbacks;

		doSomeAction1();
		rollbacks << [] {
			undoSomeAction1();
		};

		doSomeAction2();
		rollbacls << [] {
			undoSomeAction2();
		}

		// ...

		// если где-то по дороге вылетело исключение, то выполнится откат действий
		// если нет - то отменять ничего и не надо:
		rollbacks.discard()
	@endcode
*/
class Finally {
public:
	Finally() {};
	Finally(std::function<void()> fn)
	{
		_fnList.push_back(fn);
	}
	~Finally()
	{
		try
		{
			while (!_fnList.empty())
			{
				_fnList.back()();
				_fnList.pop_back();
			}
		}
		catch (...) {} // TODO: сделать трассировку
	}

	Finally(const Finally& other) = delete;
	Finally& operator=(const Finally& other) = delete;

	Finally(Finally&& other): _fnList(std::move(other._fnList)) {}
	Finally& operator=(Finally&& other) {
		_fnList = std::move(other._fnList);
		return *this;
	}

	void operator<<(std::function<void(void)> fn)
	{
		_fnList.push_back(std::move(fn));
	}

	void discard()
	{
		_fnList.clear();
	}

private:
	std::list<std::function<void(void)>> _fnList;
};

}
