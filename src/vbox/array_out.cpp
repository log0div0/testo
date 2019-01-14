
#include "array_out.hpp"
#include <string.h>

namespace vbox {

ArrayOut::~ArrayOut() {
	if (_data) {
		api->pfnArrayOutFree(_data);
	}
}

ArrayOut::ArrayOut(ArrayOut&& other): _data(other._data), _data_size(other._data_size) {
	other._data = nullptr;
	other._data_size = 0;
}

ArrayOut& ArrayOut::operator=(ArrayOut&& other) {
	std::swap(_data, other._data);
	std::swap(_data_size, other._data_size);
	return *this;
}

bool ArrayOut::operator==(const ArrayOut& other) const {
	return (_data_size == other._data_size) && (memcmp(_data, other._data, _data_size) == 0);
}

bool ArrayOut::operator!=(const ArrayOut& other) const {
	return (_data_size != other._data_size) || (memcmp(_data, other._data, _data_size) != 0);
}

const uint8_t* ArrayOut::data() const {
	return _data;
}

size_t ArrayOut::size() const {
	return _data_size;
}

const uint8_t* ArrayOut::begin() const {
	return _data;
}

const uint8_t* ArrayOut::end() const {
	return _data + _data_size;
}

uint8_t ArrayOut::operator[](size_t index) const {
	return _data[index];
}

ArrayOutIface::~ArrayOutIface() {
	if (ifaces) {
		api->pfnArrayOutFree(ifaces);
	}
}

ArrayOutIface::ArrayOutIface(ArrayOutIface&& other): ifaces(other.ifaces), ifaces_count(other.ifaces_count) {
	other.ifaces = nullptr;
	other.ifaces_count = 0;
}

ArrayOutIface& ArrayOutIface::operator=(ArrayOutIface&& other) {
	std::swap(ifaces, other.ifaces);
	std::swap(ifaces_count, other.ifaces_count);
	return *this;
}

}
