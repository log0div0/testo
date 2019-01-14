
#pragma once

#include "api.hpp"

namespace vbox {

struct ArrayOut {
	ArrayOut() = default;
	~ArrayOut();

	ArrayOut(const ArrayOut&) = delete;
	ArrayOut& operator=(const ArrayOut&) = delete;
	ArrayOut(ArrayOut&& other);
	ArrayOut& operator=(ArrayOut&& other);

	bool operator==(const ArrayOut& other) const;
	bool operator!=(const ArrayOut& other) const;

	const uint8_t* data() const;
	size_t size() const;

	const uint8_t* begin() const;
	const uint8_t* end() const;

	uint8_t operator[](size_t index) const;

private:
	uint8_t* _data = nullptr;
	ULONG _data_size = 0;

	friend struct SafeArray;
	friend struct SafeArrayView;
};

struct ArrayOutIface {
	ArrayOutIface() = default;
	~ArrayOutIface();

	ArrayOutIface(const ArrayOutIface&) = delete;
	ArrayOutIface& operator=(const ArrayOutIface&) = delete;
	ArrayOutIface(ArrayOutIface&& other);
	ArrayOutIface& operator=(ArrayOutIface&& other);

	IUnknown** ifaces = nullptr;
	ULONG ifaces_count = 0;
};

}
