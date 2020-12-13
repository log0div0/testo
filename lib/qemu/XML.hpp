
#pragma once

#include "pugixml/pugixml.hpp"

#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;

namespace vir {

struct xml_string_writer: pugi::xml_writer
{
	std::string result;

	virtual void write(const void* data, size_t size)
	{
		result.append(static_cast<const char*>(data), size);
	}
};

inline std::string node_to_string(pugi::xml_node node)
{
	xml_string_writer writer;
	node.print(writer);

	return writer.result;
}

}
