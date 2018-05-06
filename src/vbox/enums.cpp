
#include "enums.hpp"

namespace vbox {

std::ostream& operator<<(std::ostream& stream, StorageBus storage_bus) {
	switch (storage_bus) {
		case StorageBus_Null:
			return stream << "Null";
		case StorageBus_IDE:
			return stream << "IDE";
		case StorageBus_SATA:
			return stream << "SATA";
		case StorageBus_SCSI:
			return stream << "SCSI";
		case StorageBus_Floppy:
			return stream << "Floppy";
		case StorageBus_SAS:
			return stream << "SAS";
		case StorageBus_USB:
			return stream << "USB";
		case StorageBus_PCIe:
			return stream << "PCIe";
		default:
			return stream << "Unknown";
	}
}

std::ostream& operator<<(std::ostream& stream, StorageControllerType storage_controller_type) {
	switch (storage_controller_type) {
		case StorageControllerType_Null:
			return stream << "Null";
		case StorageControllerType_LsiLogic:
			return stream << "LsiLogic";
		case StorageControllerType_BusLogic:
			return stream << "BusLogic";
		case StorageControllerType_IntelAhci:
			return stream << "IntelAhci";
		case StorageControllerType_PIIX3:
			return stream << "PIIX3";
		case StorageControllerType_PIIX4:
			return stream << "PIIX4";
		case StorageControllerType_ICH6:
			return stream << "ICH6";
		case StorageControllerType_I82078:
			return stream << "I82078";
		case StorageControllerType_LsiLogicSas:
			return stream << "LsiLogicSas";
		case StorageControllerType_USB:
			return stream << "USB";
		case StorageControllerType_NVMe:
			return stream << "NVMe";
		default:
			return stream << "Unknown";
	}
}

}
