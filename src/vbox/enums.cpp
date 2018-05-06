
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

std::ostream& operator<<(std::ostream& stream, DeviceType device_type) {
	switch (device_type) {
		case DeviceType_Null:
			return stream << "Null";
		case DeviceType_Floppy:
			return stream << "Floppy";
		case DeviceType_DVD:
			return stream << "DVD";
		case DeviceType_HardDisk:
			return stream << "HardDisk";
		case DeviceType_Network:
			return stream << "Network";
		case DeviceType_USB:
			return stream << "USB";
		case DeviceType_SharedFolder:
			return stream << "SharedFolder";
		case DeviceType_Graphics3D:
			return stream << "Graphics3D";
		default:
			return stream << "Unknown";
	}
}

std::ostream& operator<<(std::ostream& stream, MediumVariant medium_variant) {
	switch (medium_variant) {
		case MediumVariant_Standard:
			return stream << "Standard";
		case MediumVariant_VmdkSplit2G:
			return stream << "VmdkSplit2G";
		case MediumVariant_VmdkRawDisk:
			return stream << "VmdkRawDisk";
		case MediumVariant_VmdkStreamOptimized:
			return stream << "VmdkStreamOptimized";
		case MediumVariant_VmdkESX:
			return stream << "VmdkESX";
		case MediumVariant_VdiZeroExpand:
			return stream << "VdiZeroExpand";
		case MediumVariant_Fixed:
			return stream << "Fixed";
		case MediumVariant_Diff:
			return stream << "Diff";
		case MediumVariant_NoCreateDir:
			return stream << "NoCreateDir";
		default:
			return stream << "Unknown";
	}
}

std::ostream& operator<<(std::ostream& stream, MediumState medium_state) {
	switch (medium_state) {
		case MediumState_NotCreated:
			return stream << "NotCreated";
		case MediumState_Created:
			return stream << "Created";
		case MediumState_LockedRead:
			return stream << "LockedRead";
		case MediumState_LockedWrite:
			return stream << "LockedWrite";
		case MediumState_Inaccessible:
			return stream << "Inaccessible";
		case MediumState_Creating:
			return stream << "Creating";
		case MediumState_Deleting:
			return stream << "Deleting";
		default:
			return stream << "Unknown";
	}
}

}
