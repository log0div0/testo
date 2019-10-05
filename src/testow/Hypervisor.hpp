
#pragma once

#include <string>
#include <memory>
#include <vector>
#include <testo/nn/Image.hpp>

struct Guest {
	Guest(const std::string& name);
	virtual ~Guest() {};

	const std::string& name() const {
		return _name;
	}

	virtual Image screenshot() = 0;

private:
	std::string _name;
};

struct Hypervisor {
	static std::shared_ptr<Hypervisor> get(const std::string& name);

	virtual ~Hypervisor() {};

	virtual std::vector<std::shared_ptr<Guest>> guests() const = 0;
};
