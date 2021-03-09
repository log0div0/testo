
#pragma once

#include <coro/StreamSocket.h>
#include <guest_additions_common_stuff/GuestAdditions.hpp>
#include <hyperv/Machine.hpp>
#ifdef WIN32
#include <hyperv/AsioWin.hpp>
#else
#include <hyperv/AsioLinux.hpp>
#endif

struct HyperVGuestAdditions: GuestAdditions {
	HyperVGuestAdditions(hyperv::Machine& domain);

private:
	virtual void send_raw(const uint8_t* data, size_t size) override;
	virtual void recv_raw(uint8_t* data, size_t size) override;

	using Socket = coro::StreamSocket<hyperv::VSocketProtocol>;

	Socket socket;
};
