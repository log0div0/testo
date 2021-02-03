
#pragma once

#include <coro/Acceptor.h>
#include <coro/StreamSocket.h>
#include <linux/vm_sockets.h>

namespace hyperv {

struct VSocketEndpoint;

struct VSocketProtocol {
	using socket = asio::basic_stream_socket<VSocketProtocol>;
	using acceptor = asio::basic_socket_acceptor<VSocketProtocol>;
	using endpoint = VSocketEndpoint;

	int family() const {
		return AF_VSOCK;
	}
	int type() const {
		return SOCK_STREAM;
	}
	int protocol() const {
		return 0;
	}
};

struct VSocketEndpoint {
	VSocketEndpoint(): sa_() {
		sa_.svm_family = AF_VSOCK;
	}

	VSocketEndpoint(unsigned int port, unsigned int cid = VMADDR_CID_ANY): VSocketEndpoint() {
		sa_.svm_port = port;
		sa_.svm_cid = cid;
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
	sockaddr_vm sa_;
};

using Acceptor = coro::Acceptor<hyperv::VSocketProtocol>;
using Socket = coro::StreamSocket<hyperv::VSocketProtocol>;

}
