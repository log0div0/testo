
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
	bool cableConnected() const;
	void setEnabled(bool is_enabled);
	bool enabled() const;
	void setInternalNetwork(const std::string& network);
	void setAttachmentType(NetworkAttachmentType type);
	void setAdapterType(NetworkAdapterType type);
	void setMAC(const std::string& mac);

	INetworkAdapter* handle = nullptr;
};

}