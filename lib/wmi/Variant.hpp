
#pragma once

#include <Windows.h>
#include <oaidl.h>
#include <string>
#include <vector>

namespace wmi {

struct Variant: VARIANT {
	Variant();
	Variant(bool value);
	Variant(const char* str);
	Variant(const std::string& str);
	Variant(const std::vector<uint8_t>& buffer);
	Variant(const std::vector<std::string>& strs);
	Variant(uint16_t value);
	Variant(int16_t value);
	Variant(uint32_t value);
	Variant(int32_t value);
	Variant(uint64_t value);
	Variant(int64_t value);
	~Variant();

	Variant(const Variant& other) = delete;
	Variant& operator=(const Variant& other) = delete;
	Variant(Variant&& other);
	Variant& operator=(Variant&& other);

	operator std::string() const;
	operator std::vector<uint8_t>() const;
	operator std::vector<std::string>() const;

	operator uint16_t() const;
	operator int16_t() const;
	operator uint32_t() const;
	operator int32_t() const;

	template <typename T>
	T get() const {
		return *this;
	}

private:
	void check_type(VARENUM expected) const;
};

}
