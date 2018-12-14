
#include "enums.hpp"
#include <vector>

namespace vbox {

struct Snapshot {
	Snapshot(ISnapshot* handle);
	~Snapshot();
	Snapshot(const Snapshot& other) = delete;
	Snapshot& operator=(const Snapshot& other) = delete;

	Snapshot(Snapshot&& other);
	Snapshot& operator=(Snapshot&& other);

	std::string name() const;
	std::string id() const;
	std::string getDescription() const;
	void setDescription(const std::string& description) const;
	std::vector<Snapshot> children() const;

	ISnapshot* handle;
};
	
}
