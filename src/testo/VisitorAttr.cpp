
#include "VisitorAttr.hpp"
#include "Exceptions.hpp"
#include "TemplateLiterals.hpp"

static uint32_t size_to_mb(const std::string& size) {
	uint32_t result = std::stoul(size.substr(0, size.length() - 2));
	if (size[size.length() - 2] == 'M') {
		result = result * 1;
	} else if (size[size.length() - 2] == 'G') {
		result = result * 1024;
	} else {
		throw Exception("Unknown size specifier"); //should not happen ever
	}

	return result;
}

static bool str_to_bool(const std::string& str) {
	if (str == "true") {
		return true;
	} else if (str == "false") {
		return false;
	} else {
		throw std::runtime_error("Can't convert \"" + str + "\" to boolean");
	}
}

struct UnknownAttributeError: Exception {
	UnknownAttributeError(std::shared_ptr<AST::Attr> attr):
		Exception(std::string(attr->begin()) + ": Error: unknown attribute name: \"" + attr->name() + "\"") {}
};

template <typename Visitor>
nlohmann::json VisitorAttr::visit_attr_block(std::shared_ptr<AST::Attr> attr, Visitor&& visitor) {
	if (!attr->id) {
		throw Exception(std::string(attr->end()) + ": Error: attribute \"" + attr->name() +
			"\" requires a name");
	}
	auto p = std::dynamic_pointer_cast<AST::AttrBlock>(attr->value);
	if (!p) {
		throw Exception(std::string(attr->begin()) + ": Error: attribute \"" + attr->name() + "\" is expected to be a block of attibutes");
	}
	return visit_attr_block(p, visitor);
}

template <typename Visitor>
nlohmann::json VisitorAttr::visit_attr_block(std::shared_ptr<AST::AttrBlock> attr_block, Visitor&& visitor) {
	nlohmann::json config;
	for (auto attr: attr_block->attrs) {
		if (config.count(attr->name())) {
			if (!config.at(attr->name()).is_array()) {
				throw Exception(std::string(attr->begin()) + ": Error: duplicate attribute: \"" + attr->name() + "\"");
			}
		}
		nlohmann::json j = visitor(attr);
		if (attr->id) {
			j["name"] = attr->id.value();
			config[attr->name()].push_back(j);
		}  else {
			config[attr->name()] = j;
		}
	}
	return config;
}

nlohmann::json VisitorAttr::visit_attr_simple_value(std::shared_ptr<AST::Attr> attr, Token::category category) {
	if (attr->id) {
		throw Exception(std::string(attr->end()) + ": Error: attribute \"" + attr->name() +
			"\" must have no name");
	}
	auto p = std::dynamic_pointer_cast<AST::AttrSimpleValue>(attr->value);
	if (!p) {
		throw Exception(std::string(attr->begin()) + ": Error: attribute \"" + attr->name() + "\" is expected to be a simple attibute");
	}
	return visit_attr_simple_value(p, category);
}

nlohmann::json VisitorAttr::visit_attr_simple_value(std::shared_ptr<AST::AttrSimpleValue> p, Token::category category) {
	if (category == Token::category::quoted_string) {
		try {
			return template_literals::Parser().resolve(p->value->text(), stack);
		} catch (const std::exception& error) {
			std::throw_with_nested(ResolveException(p->value->begin(), p->value->text()));
		}
	} else {
		p->value->expected_token_type = category;
		std::string value;
		try {
			value = IR::StringTokenUnion(p->value, stack).resolve();
		} catch (const std::exception& error) {
			throw Exception(std::string(p->begin()) + ": Error: can't convert value string \"" +
				p->value->text() + "\" into expected \"" + Token::type_to_string(category) + "\"");
		}

		if (category == Token::category::number) {
			if (std::stoi(value) < 0) {
				throw Exception(std::string(p->begin()) + ": Error: numeric attr can't be negative: " + value);
			}
			return std::stoul(value);
		} else if (category == Token::category::size) {
			return size_to_mb(value);
		} else if (category == Token::category::boolean) {
			return str_to_bool(value);
		} else {
			throw Exception(std::string(p->begin()) + ": Error: unsupported attr: " + value);
		}
	}
}

nlohmann::json VisitorAttrMachine::visit(std::shared_ptr<AST::AttrBlock> attr_block) {
	return visit_attr_block(attr_block, [&](auto attr) {
		if (attr->name() == "ram") {
			return visit_attr_simple_value(attr, Token::category::size);
		} else if (attr->name() == "iso") {
			return visit_attr_simple_value(attr, Token::category::quoted_string);
		} else if (attr->name() == "nic") {
			return visit_nic(attr);
		} else if (attr->name() == "disk") {
			return visit_disk(attr);
		} else if (attr->name() == "video") {
			return visit_video(attr);
		} else if (attr->name() == "shared_folder") {
			return visit_shared_folder(attr);
		} else if (attr->name() == "cpus") {
			return visit_attr_simple_value(attr, Token::category::number);
		} else if (attr->name() == "qemu_spice_agent") {
			return visit_attr_simple_value(attr, Token::category::boolean);
		} else if (attr->name() == "qemu_enable_usb3") {
			return visit_attr_simple_value(attr, Token::category::boolean);
		} else if (attr->name() == "loader") {
			return visit_attr_simple_value(attr, Token::category::quoted_string);
		} else {
			throw UnknownAttributeError(attr);
		}
	});
}

nlohmann::json VisitorAttrMachine::visit_disk(std::shared_ptr<AST::Attr> attr) {
	return visit_attr_block(attr, [&](auto attr) {
		if (attr->name() == "size") {
			return visit_attr_simple_value(attr, Token::category::size);
		} else if (attr->name() == "source") {
			return visit_attr_simple_value(attr, Token::category::quoted_string);
		} else {
			throw UnknownAttributeError(attr);
		}
	});
}

nlohmann::json VisitorAttrMachine::visit_nic(std::shared_ptr<AST::Attr> attr) {
	return visit_attr_block(attr, [&](auto attr) {
		if (attr->name() == "attached_to") {
			return visit_attr_simple_value(attr, Token::category::quoted_string);
		} else if (attr->name() == "attached_to_dev") {
			return visit_attr_simple_value(attr, Token::category::quoted_string);
		} else if (attr->name() == "mac") {
			return visit_attr_simple_value(attr, Token::category::quoted_string);
		} else if (attr->name() == "adapter_type") {
			return visit_attr_simple_value(attr, Token::category::quoted_string);
		} else {
			throw UnknownAttributeError(attr);
		}
	});
}

nlohmann::json VisitorAttrMachine::visit_video(std::shared_ptr<AST::Attr> attr) {
	return visit_attr_block(attr, [&](auto attr) {
		if (attr->name() == "qemu_mode") {
			return visit_attr_simple_value(attr, Token::category::quoted_string); // deprecated
		} else if (attr->name() == "adapter_type") {
			return visit_attr_simple_value(attr, Token::category::quoted_string);
		} else {
			throw UnknownAttributeError(attr);
		}
	});
}

nlohmann::json VisitorAttrMachine::visit_shared_folder(std::shared_ptr<AST::Attr> attr) {
	return visit_attr_block(attr, [&](auto attr) {
		if (attr->name() == "host_path") {
			return visit_attr_simple_value(attr, Token::category::quoted_string);
		} else if (attr->name() == "readonly") {
			return visit_attr_simple_value(attr, Token::category::boolean);
		} else {
			throw UnknownAttributeError(attr);
		}
	});
}

nlohmann::json VisitorAttrFlashDrive::visit(std::shared_ptr<AST::AttrBlock> attr_block) {
	return visit_attr_block(attr_block, [&](auto attr) {
		if (attr->name() == "fs") {
			return visit_attr_simple_value(attr, Token::category::quoted_string);
		} if (attr->name() == "size") {
			return visit_attr_simple_value(attr, Token::category::size);
		} if (attr->name() == "folder") {
			return visit_attr_simple_value(attr, Token::category::quoted_string);
		} else {
			throw UnknownAttributeError(attr);
		}
	});
}

nlohmann::json VisitorAttrNetwork::visit(std::shared_ptr<AST::AttrBlock> attr_block) {
	return visit_attr_block(attr_block, [&](auto attr) {
		if (attr->name() == "mode") {
			return visit_attr_simple_value(attr, Token::category::quoted_string);
		} else {
			throw UnknownAttributeError(attr);
		}
	});
}

nlohmann::json VisitorAttrTest::visit(std::shared_ptr<AST::AttrBlock> attr_block) {
	return visit_attr_block(attr_block, [&](auto attr) {
		if (attr->name() == "no_snapshots") {
			return visit_attr_simple_value(attr, Token::category::boolean);
		} if (attr->name() == "description") {
			return visit_attr_simple_value(attr, Token::category::quoted_string);
		} else {
			throw UnknownAttributeError(attr);
		}
	});
}
