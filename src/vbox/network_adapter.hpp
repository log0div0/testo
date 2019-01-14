
#pragma once

#include "enums.hpp"
#include <unordered_map>
#include <vector>

namespace vbox {

struct NetworkAdapter {
	NetworkAdapter(INetworkAdapter* handle);
	~NetworkAdapter();

	NetworkAdapter(const NetworkAdapter&) = delete;
	NetworkAdapter& operator=(const NetworkAdapter&) = delete;
	NetworkAdapter(NetworkAdapter&& other);
	NetworkAdapter& operator=(NetworkAdapter&& other);

	void setCableConnected(bool is_connected);

	void setEnabled(bool is_enabled) const;
	void setInternalNetwork(const std::string& network) const;
	void setAttachmentType(NetworkAttachmentType type) const;
	void setAdapterType(NetworkAdapterType type) const;
	void setMAC(const std::string& mac);

	INetworkAdapter* handle = nullptr;
};

}