
#include "Homm3Object.hpp"

namespace nn {

const std::vector<std::string> Homm3Object::classes_names = {
	"cancel",
	"cursed_temple",
	"garden_of_revelation",
	"hall",
	"hero",
	"hovel",
	"mage_guild",
	"magic_spring",
	"ok",
	"peasant",
	"redwood_observatory",
	"scholar",
	"sign",
	"skeleton",
	"spell_destroy_undead",
	"spell_slow",
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
