
#pragma once

#include "API.hpp"
#include "Register.hpp"

struct Environment {
	virtual ~Environment() = default;

	virtual void setup() = 0;
	virtual void cleanup() = 0;
};

struct VboxEnvironment: public Environment {
	VboxEnvironment(): api(API::instance()) {}
	~VboxEnvironment();

	void setup() override;
	void cleanup() override;

	API& api;
};
