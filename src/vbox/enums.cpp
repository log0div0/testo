
#include "enums.hpp"

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
	stream << "{";
	size_t counter = 0;
	for (size_t i = 0; i < sizeof(int) * 8; ++i) {
		int bit = 1 << i;
		if (medium_variant & bit) {
			if (counter != 0) {
				stream << "|";
			}
			switch (bit) {
				case MediumVariant_VmdkSplit2G:
					stream << "VmdkSplit2G";
					break;
				case MediumVariant_VmdkRawDisk:
					stream << "VmdkRawDisk";
					break;
				case MediumVariant_VmdkStreamOptimized:
					stream << "VmdkStreamOptimized";
					break;
				case MediumVariant_VmdkESX:
					stream << "VmdkESX";
					break;
				case MediumVariant_VdiZeroExpand:
					stream << "VdiZeroExpand";
					break;
				case MediumVariant_Fixed:
					stream << "Fixed";
					break;
				case MediumVariant_Diff:
					stream << "Diff";
					break;
				case MediumVariant_NoCreateDir:
					stream << "NoCreateDir";
					break;
				default:
					stream << "Unknown";
					break;
			}
			++counter;
		}
	}
	stream << "}";
	return stream;
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

std::ostream& operator<<(std::ostream& stream, MachineState machine_state) {
	switch (machine_state) {
		case MachineState_Null:
			return stream << "Null";
		case MachineState_PoweredOff:
			return stream << "PoweredOff";
		case MachineState_Saved:
			return stream << "Saved";
		case MachineState_Teleported:
			return stream << "Teleported";
		case MachineState_Aborted:
			return stream << "Aborted";
		case MachineState_Running:
			return stream << "Running";
		case MachineState_Paused:
			return stream << "Paused";
		case MachineState_Stuck:
			return stream << "Stuck";
		case MachineState_Teleporting:
			return stream << "Teleporting";
		case MachineState_LiveSnapshotting:
			return stream << "LiveSnapshotting";
		case MachineState_Starting:
			return stream << "Starting";
		case MachineState_Stopping:
			return stream << "Stopping";
		case MachineState_Saving:
			return stream << "Saving";
		case MachineState_Restoring:
			return stream << "Restoring";
		case MachineState_TeleportingPausedVM:
			return stream << "TeleportingPausedVM";
		case MachineState_TeleportingIn:
			return stream << "TeleportingIn";
		case MachineState_FaultTolerantSyncing:
			return stream << "FaultTolerantSyncing";
		case MachineState_DeletingSnapshotOnline:
			return stream << "DeletingSnapshotOnline";
		case MachineState_DeletingSnapshotPaused:
			return stream << "DeletingSnapshotPaused";
		case MachineState_OnlineSnapshotting:
			return stream << "OnlineSnapshotting";
		case MachineState_RestoringSnapshot:
			return stream << "RestoringSnapshot";
		case MachineState_DeletingSnapshot:
			return stream << "DeletingSnapshot";
		case MachineState_SettingUp:
			return stream << "SettingUp";
		case MachineState_Snapshotting:
			return stream << "Snapshotting";
		default:
			return stream << "Unknown";
	}
}

std::ostream& operator<<(std::ostream& stream, SessionState session_state) {
	switch (session_state) {
		case SessionState_Null:
			return stream << "Null";
		case SessionState_Unlocked:
			return stream << "Unlocked";
		case SessionState_Locked:
			return stream << "Locked";
		case SessionState_Spawning:
			return stream << "Spawning";
		case SessionState_Unlocking:
			return stream << "Unlocking";
		default:
			return stream << "Unknown";
	}
}
