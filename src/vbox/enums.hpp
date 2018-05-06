
#pragma once

#include <VBoxCAPI/VBoxCAPI_v5_2.h>
#include <ostream>

namespace vbox {

std::ostream& operator<<(std::ostream& stream, StorageBus storage_bus);
std::ostream& operator<<(std::ostream& stream, StorageControllerType storage_controller_type);
std::ostream& operator<<(std::ostream& stream, DeviceType device_type);
std::ostream& operator<<(std::ostream& stream, MediumVariant medium_variant);

}