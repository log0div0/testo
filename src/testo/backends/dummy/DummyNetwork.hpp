
#pragma once

#include "../Network.hpp"

struct DummyNetwork: Network {
	DummyNetwork() = delete;
	DummyNetwork(const DummyNetwork& other) = delete;
	DummyNetwork(const nlohmann::json& config): Network(config) {}
	~DummyNetwork() {}

	bool is_defined() override;
	void create() override;
	void undefine() override;

private:
	fs::path metadata_file() const {
		fs::path result = "./dummy_hypervisor_files";
		result = result / id();
		return result;
	};
};
