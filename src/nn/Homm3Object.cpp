
#include "Homm3Object.hpp"

namespace nn {

const std::vector<std::string> Homm3Object::classes_names = {
	"garden_of_revelation",
	"hero",
	"hovel",
	"magic_spring",
	"peasant",
	"redwood_observatory",
	"scholar",
	"sign",
	"skeleton",
	"star_axis",
	"town",
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
