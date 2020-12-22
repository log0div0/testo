
#include "Homm3Object.hpp"

namespace nn {

const std::vector<std::string> Homm3Object::classes_names = {
	"hero",
	"town"
};

bool Homm3Object::check_class_name(const std::string& name) {
	for (auto& x: classes_names) {
		if (x == name) {
			return true;
		}
	}
	return false;
}

}
