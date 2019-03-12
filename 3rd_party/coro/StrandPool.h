
#pragma once

#include "coro/Strand.h"
#include "coro/Coro.h"
#include <map>

namespace coro {

/// Класс для иерархического управления корутинами
class StrandPool {
public:
	/// Блокирует поток выполнения до тех пор, пока не завершатся все дочерние корутины
	~StrandPool();
	/// Запустить новую корутину в отдельном Strand
	std::shared_ptr<Strand> exec(std::function<void()> routine);
	/// Дождаться завершения всех дочерних корутин
	void waitAll(bool noThrow);
	/// Отменить все дочерние корутины
	void cancelAll();

private:
	void onStrandDone(std::shared_ptr<Strand> strand);

	std::string token() const;

	Strand* _parentStrand = Strand::current();
	Coro* _parentCoro = Coro::current();
	std::map<std::shared_ptr<Strand>, std::shared_ptr<Coro>> _childStrands;
};

}