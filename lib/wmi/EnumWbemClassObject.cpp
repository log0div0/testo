
#include "EnumWbemClassObject.hpp"
#include "Error.hpp"

namespace wmi {

std::vector<WbemClassObject> EnumWbemClassObject::next(size_t count) {
	try {
		ULONG returned = 0;
		std::vector<IWbemClassObject*> buffer(count, nullptr);
		HRESULT hres = handle->Next(WBEM_INFINITE, count, buffer.data(), &returned);
		std::vector<WbemClassObject> result(buffer.begin(), buffer.begin() + returned);
		throw_if_failed(hres);
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::vector<WbemClassObject> EnumWbemClassObject::getAll(size_t batch_size) {
	try {
		std::vector<wmi::WbemClassObject> result;
		while (true) {
			std::vector<wmi::WbemClassObject> objects = next(batch_size);
			if (objects.size() == 0) {
				break;
			}
			for (auto& object: objects) {
				result.push_back(std::move(object));
			}
		}
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

WbemClassObject EnumWbemClassObject::getOne() {
	try {
		auto objects = getAll();
		if (objects.size() == 0) {
			throw std::runtime_error("No object found");
		}
		if (objects.size() != 1) {
			throw std::runtime_error("Found more objects than expected");
		}
		return std::move(objects.at(0));
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
