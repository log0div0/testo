
#pragma once

#include <coro/Acceptor.h>
#include <coro/StreamSocket.h>
#include <ws2def.h>
#include <hvsocket.h>

namespace hyperv {

struct VSocketEndpoint;

struct VSocketProtocol {
	using socket = asio::basic_stream_socket<VSocketProtocol>;
	using acceptor = asio::basic_socket_acceptor<VSocketProtocol>;
	using endpoint = VSocketEndpoint;

	int family() const {
		return AF_HYPERV;
	}
	int type() const {
		return SOCK_STREAM;
	}
	int protocol() const {
		return HV_PROTOCOL_RAW;
	}
};

struct VSocketEndpoint {
	VSocketEndpoint(): sa_() {
		sa_.Family = AF_HYPERV;
	}

	VSocketEndpoint(GUID serviceId, GUID vmId = HV_GUID_WILDCARD): VSocketEndpoint() {
		sa_.ServiceId = serviceId;
		sa_.VmId = vmId;
	}

	sockaddr* data() {
		return (sockaddr*)&sa_;
	}

	const sockaddr* data() const {
		return (const sockaddr*)&sa_;
	}

	size_t size() const {
		return sizeof(sa_);
	}

	size_t capacity() const {
		return sizeof(sa_);
	}

	void resize(size_t size) {
	}

	VSocketProtocol protocol() const {
		return VSocketProtocol();
	}

private:
	SOCKADDR_HV sa_;
};

}
