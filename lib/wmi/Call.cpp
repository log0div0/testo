
#include "Call.hpp"
#include "Error.hpp"
#include "WbemObjectTextSrc.hpp"
#include <chrono>
#include <thread>

namespace wmi {

Call::Call(WbemServices services_, std::string class_name_, std::string method_name_):
	services(std::move(services_)), class_name(std::move(class_name_)), method_name(std::move(method_name_))
{
	try {
		auto method = services.getObject(class_name).getMethod(method_name);
		if (method.handle) {
			method_instance = method.spawnInstance();
		}
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Call& Call::with(std::string name, const Variant& value) {
	try {
		method_instance.put(name, value);
		return *this;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Call& Call::with(std::string name, const WbemClassObject& object) {
	try {
		return with(name, WbemObjectTextSrc().getText(object));
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Call& Call::with(std::string name, const std::vector<WbemClassObject>& objects) {
	try {
		std::vector<std::string> strings;
		WbemObjectTextSrc objectTextSrc;
		for (auto& object: objects) {
			strings.push_back(objectTextSrc.getText(object));
		}
		return with(name, strings);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

WbemClassObject Call::exec(WbemClassObject object) {
	try {
		auto result = services.execMethod(object.path(), method_name, method_instance);
		int32_t returnValue = result.get("ReturnValue");
		if (returnValue != 0) {
			if (returnValue == 4096) {
				std::string jobRef = result.get("Job");
				while (true) {
					auto job = services.getObject(jobRef);
					int32_t jobState = job.get("JobState");
					if (jobState == 4) {
						std::this_thread::sleep_for(std::chrono::milliseconds(100));
						continue;
					}
					if (jobState == 7) {
						break;
					}
					std::string errorDescription = job.get("ErrorDescription");
					throw std::runtime_error(errorDescription);
				}
			} else {
				throw std::runtime_error("ReturnValue == " + std::to_string(returnValue));
			}
		}
		return result;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

WbemClassObject Call::exec() {
	try {
		return exec(services.execQuery("SELECT * FROM " + class_name).getOne());
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
