
#pragma once

#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iterator>
#include <vector>

namespace inipp
{

struct inisection: std::map<std::string,std::string> {
	inisection(const std::string& name = ""): _name(name) {}

	const std::string& name() const {
		return _name;
	}

	const std::string& get(const std::string& key, const std::string& default_value) const {
		auto it = find(key);
		if (it != end()) {
			return it->second;
		} else {
			return default_value;
		}
	}

	int get_int(const std::string& key, int default_value) const {
		auto it = find(key);
		if (it != end()) {
			return std::stoi(it->second);
		} else {
			return default_value;
		}
	}

	float get_float(const std::string& key, float default_value) const {
		auto it = find(key);
		if (it != end()) {
			return std::stof(it->second);
		} else {
			return default_value;
		}
	}

	const std::string& get(const std::string& key) const {
		auto it = find(key);
		if (it != end()) {
			return it->second;
		}
		throw std::runtime_error("Cannot find key " + key + " in section " + _name);
	}

	int get_int(const std::string& key) const {
		auto it = find(key);
		if (it != end()) {
			return std::stoi(it->second);
		}
		throw std::runtime_error("Cannot find key " + key + " in section " + _name);
	}

	float get_float(const std::string& key) const {
		auto it = find(key);
		if (it != end()) {
			return std::stof(it->second);
		}
		throw std::runtime_error("Cannot find key " + key + " in section " + _name);
	}

private:
	std::string _name;
};

inline std::string trim(const std::string& str, const std::string& whitespace = " \t\n\r\f\v") {
	size_t startpos = str.find_first_not_of(whitespace);
	size_t endpos = str.find_last_not_of(whitespace);

	// only whitespace, return empty line
	if(startpos == std::string::npos || endpos == std::string::npos) {
		return std::string();
	}

	// trim leading and trailing whitespace
	return str.substr(startpos, endpos - startpos + 1);
}

inline bool split(const std::string& in, const std::string& sep, std::string& first, std::string& second) {
	size_t eqpos = in.find(sep);

	if(eqpos == std::string::npos) {
		return false;
	}

	first = in.substr(0, eqpos);
	second = in.substr(eqpos + sep.size(), in.size() - eqpos - sep.size());

	return true;
}

struct syntax_error : public std::runtime_error
{
public:
	syntax_error(const std::string& msg)
		: std::runtime_error(msg)
	{ /* empty */ };
};

struct inifile: inisection
{
public:
	inifile(std::ifstream& infile) {
		inisection* cursec = this;
		std::string line;

		while(std::getline(infile, line)) {
			// trim line
			line = trim(line);

			// ignore empty lines and comments
			if(line.empty() || line[0] == '#') {
				continue;
			}

			// section?
			if(line[0] == '[') {
				if(line[line.size() - 1] != ']') {
					throw syntax_error("The section '" + line +
									 "' is missing a closing bracket.");
				}

				line = trim(line.substr(1, line.size() - 2));
				_sections.emplace_back(line);
				cursec = &_sections.back();
				continue;
			}

			// entry: split by "=", trim and set
			std::string key;
			std::string value;

			if(split(line, "=", key, value)) {
				(*cursec)[trim(key)] = trim(value);
				continue;
			}

			// throw exception on invalid line
			throw syntax_error("The line '" + line + "' is invalid.");
		}
	}

	const std::vector<inisection>& sections() const {
		return _sections;
	}

protected:
	std::vector<inisection> _sections;
};

}
