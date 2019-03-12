
#include "coro/StrandPool.h"
#include <coro/Finally.h>

namespace coro {

StrandPool::~StrandPool() {
	cancelAll();
	waitAll(true);
}

std::shared_ptr<Strand> StrandPool::exec(std::function<void()> routine) {
	auto strand = std::make_shared<Strand>();
	auto coro = std::make_shared<Coro>([=] {
		Finally cleanup([=] {
			strand->post([=] {
				onStrandDone(strand);
			});
		});

		routine();
	});
	_childStrands.emplace(strand, coro);
	strand->post([=] {
		coro->start();
	});
	return strand;
}

void StrandPool::waitAll(bool noThrow) {
	if (_childStrands.empty()) {
		return;
	}

	if (noThrow) {
		_parentCoro->yield({token()});
	} else {
		_parentCoro->yield({token(), TokenThrow});
	}

	assert(_childStrands.empty());
}

void StrandPool::cancelAll() {
	for (auto it: _childStrands) {
		it.first->post([=] {
			it.second->cancel();
		});
	}
}

void StrandPool::onStrandDone(std::shared_ptr<Strand> childStrand) {
	_parentStrand->post([=] {
		_childStrands.erase(childStrand);

		if (_childStrands.empty()) {
			_parentCoro->resume(token());
		}
	});
}

std::string StrandPool::token() const {
	return "StrandPool " + std::to_string((uint64_t)this);
}

}