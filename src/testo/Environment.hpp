
#pragma once

#include "API.hpp"
#include "Register.hpp"

struct Environment {
	Environment(): api(API::instance()) {}
	~Environment();

	void setup();
	void cleanup();

	API& api;
};
