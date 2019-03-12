
#pragma once

#include "coro/IoService.h"

namespace coro {

class Strand;

extern thread_local Strand* t_strand;

/// Вспомогательный класс для Strand
template <typename T>
struct WrappedHandler {
	WrappedHandler(Strand* strand, T&& t): _strand(strand), _t(std::move(t)) {}

	template <typename ...Args>
	void operator()(Args&&... args) {
		t_strand = _strand;
		_t(std::forward<Args>(args)...);
		t_strand = nullptr;
	}

	Strand* _strand;
	T _t;
};

/// Wrapper вокруг asio::io_service::strand
class Strand {
public:
	static Strand* current();

	Strand(IoService* ioService = IoService::current()): _impl(*ioService) {}

	template <typename T>
	auto wrap(T&& t) {
		return _impl.wrap(WrappedHandler<T>(this, std::forward<T>(t)));
	}

	template <typename T>
	void post(T&& t) {
		_impl.post(WrappedHandler<T>(this, std::forward<T>(t)));
	}

	template <typename T>
	void dispatch(T&& t) {
		_impl.dispatch(WrappedHandler<T>(this, std::forward<T>(t)));
	}

	operator const asio::io_service::strand&() const { return _impl; }
	operator asio::io_service::strand&() { return _impl; }

private:
	asio::io_service::strand _impl;
};

}