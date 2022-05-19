
#include "QemuVM.hpp"
#include "QemuFlashDrive.hpp"
#include "QemuGuestAdditions.hpp"
#include "QemuEnvironment.hpp"

#include <coro/Timer.h>
#include <coro/Timeout.h>

#include <os/Process.hpp>

#include <fmt/format.h>
#include <thread>

using namespace std::chrono_literals;

void create_img_with_backing_file(const fs::path& img, const fs::path& backing_file) {
	os::Process::exec(fmt::format("qemu-img create -b \"{}\" -f qcow2 -F qcow2 \"{}\"", backing_file.generic_string(), img.generic_string()));
}

void create_empty_img(const fs::path& img, uint32_t size_in_megabytes) {
	os::Process::exec(fmt::format("qemu-img create -f qcow2 \"{}\" {}M", img.generic_string(), size_in_megabytes));
}

const std::unordered_map<uint16_t, uint16_t> virKeyCodeTable_rfb = {
  {0x1, 0x1}, /* KEY_ESC */
  {0x2, 0x2}, /* KEY_1 */
  {0x3, 0x3}, /* KEY_2 */
  {0x4, 0x4}, /* KEY_3 */
  {0x5, 0x5}, /* KEY_4 */
  {0x6, 0x6}, /* KEY_5 */
  {0x7, 0x7}, /* KEY_6 */
  {0x8, 0x8}, /* KEY_7 */
  {0x9, 0x9}, /* KEY_8 */
  {0xa, 0xa}, /* KEY_9 */
  {0xb, 0xb}, /* KEY_0 */
  {0xc, 0xc}, /* KEY_MINUS */
  {0xd, 0xd}, /* KEY_EQUAL */
  {0xe, 0xe}, /* KEY_BACKSPACE */
  {0xf, 0xf}, /* KEY_TAB */
  {0x10, 0x10}, /* KEY_Q */
  {0x11, 0x11}, /* KEY_W */
  {0x12, 0x12}, /* KEY_E */
  {0x13, 0x13}, /* KEY_R */
  {0x14, 0x14}, /* KEY_T */
  {0x15, 0x15}, /* KEY_Y */
  {0x16, 0x16}, /* KEY_U */
  {0x17, 0x17}, /* KEY_I */
  {0x18, 0x18}, /* KEY_O */
  {0x19, 0x19}, /* KEY_P */
  {0x1a, 0x1a}, /* KEY_LEFTBRACE */
  {0x1b, 0x1b}, /* KEY_RIGHTBRACE */
  {0x1c, 0x1c}, /* KEY_ENTER */
  {0x1d, 0x1d}, /* KEY_LEFTCTRL */
  {0x1e, 0x1e}, /* KEY_A */
  {0x1f, 0x1f}, /* KEY_S */
  {0x20, 0x20}, /* KEY_D */
  {0x21, 0x21}, /* KEY_F */
  {0x22, 0x22}, /* KEY_G */
  {0x23, 0x23}, /* KEY_H */
  {0x24, 0x24}, /* KEY_J */
  {0x25, 0x25}, /* KEY_K */
  {0x26, 0x26}, /* KEY_L */
  {0x27, 0x27}, /* KEY_SEMICOLON */
  {0x28, 0x28}, /* KEY_APOSTROPHE */
  {0x29, 0x29}, /* KEY_GRAVE */
  {0x2a, 0x2a}, /* KEY_LEFTSHIFT */
  {0x2b, 0x2b}, /* KEY_BACKSLASH */
  {0x2c, 0x2c}, /* KEY_Z */
  {0x2d, 0x2d}, /* KEY_X */
  {0x2e, 0x2e}, /* KEY_C */
  {0x2f, 0x2f}, /* KEY_V */
  {0x30, 0x30}, /* KEY_B */
  {0x31, 0x31}, /* KEY_N */
  {0x32, 0x32}, /* KEY_M */
  {0x33, 0x33}, /* KEY_COMMA */
  {0x34, 0x34}, /* KEY_DOT */
  {0x35, 0x35}, /* KEY_SLASH */
  {0x36, 0x36}, /* KEY_RIGHTSHIFT */
  {0x37, 0x37}, /* KEY_KPASTERISK */
  {0x38, 0x38}, /* KEY_LEFTALT */
  {0x39, 0x39}, /* KEY_SPACE */
  {0x3a, 0x3a}, /* KEY_CAPSLOCK */
  {0x3b, 0x3b}, /* KEY_F1 */
  {0x3c, 0x3c}, /* KEY_F2 */
  {0x3d, 0x3d}, /* KEY_F3 */
  {0x3e, 0x3e}, /* KEY_F4 */
  {0x3f, 0x3f}, /* KEY_F5 */
  {0x40, 0x40}, /* KEY_F6 */
  {0x41, 0x41}, /* KEY_F7 */
  {0x42, 0x42}, /* KEY_F8 */
  {0x43, 0x43}, /* KEY_F9 */
  {0x44, 0x44}, /* KEY_F10 */
  {0x45, 0x45}, /* KEY_NUMLOCK */
  {0x46, 0x46}, /* KEY_SCROLLLOCK */
  {0x47, 0x47}, /* KEY_KP7 */
  {0x48, 0x48}, /* KEY_KP8 */
  {0x49, 0x49}, /* KEY_KP9 */
  {0x4a, 0x4a}, /* KEY_KPMINUS */
  {0x4b, 0x4b}, /* KEY_KP4 */
  {0x4c, 0x4c}, /* KEY_KP5 */
  {0x4d, 0x4d}, /* KEY_KP6 */
  {0x4e, 0x4e}, /* KEY_KPPLUS */
  {0x4f, 0x4f}, /* KEY_KP1 */
  {0x50, 0x50}, /* KEY_KP2 */
  {0x51, 0x51}, /* KEY_KP3 */
  {0x52, 0x52}, /* KEY_KP0 */
  {0x53, 0x53}, /* KEY_KPDOT */
  {0x54, 0x54}, /* unnamed */
  {0x55, 0x76}, /* KEY_ZENKAKUHANKAKU */
  {0x56, 0x56}, /* KEY_102ND */
  {0x57, 0x57}, /* KEY_F11 */
  {0x58, 0x58}, /* KEY_F12 */
  {0x59, 0x73}, /* KEY_RO */
  {0x5a, 0x78}, /* KEY_KATAKANA */
  {0x5b, 0x77}, /* KEY_HIRAGANA */
  {0x5c, 0x79}, /* KEY_HENKAN */
  {0x5d, 0x70}, /* KEY_KATAKANAHIRAGANA */
  {0x5e, 0x7b}, /* KEY_MUHENKAN */
  {0x5f, 0x5c}, /* KEY_KPJPCOMMA */
  {0x60, 0x9c}, /* KEY_KPENTER */
  {0x61, 0x9d}, /* KEY_RIGHTCTRL */
  {0x62, 0xb5}, /* KEY_KPSLASH */
  {0x63, 0x54}, /* KEY_SYSRQ */
  {0x64, 0xb8}, /* KEY_RIGHTALT */
  {0x65, 0x5b}, /* KEY_LINEFEED */
  {0x66, 0xc7}, /* KEY_HOME */
  {0x67, 0xc8}, /* KEY_UP */
  {0x68, 0xc9}, /* KEY_PAGEUP */
  {0x69, 0xcb}, /* KEY_LEFT */
  {0x6a, 0xcd}, /* KEY_RIGHT */
  {0x6b, 0xcf}, /* KEY_END */
  {0x6c, 0xd0}, /* KEY_DOWN */
  {0x6d, 0xd1}, /* KEY_PAGEDOWN */
  {0x6e, 0xd2}, /* KEY_INSERT */
  {0x6f, 0xd3}, /* KEY_DELETE */
  {0x70, 0xef}, /* KEY_MACRO */
  {0x71, 0xa0}, /* KEY_MUTE */
  {0x72, 0xae}, /* KEY_VOLUMEDOWN */
  {0x73, 0xb0}, /* KEY_VOLUMEUP */
  {0x74, 0xde}, /* KEY_POWER */
  {0x75, 0x59}, /* KEY_KPEQUAL */
  {0x76, 0xce}, /* KEY_KPPLUSMINUS */
  {0x77, 0xc6}, /* KEY_PAUSE */
  {0x78, 0x8b}, /* KEY_SCALE */
  {0x79, 0x7e}, /* KEY_KPCOMMA */
  {0x7b, 0x8d}, /* KEY_HANJA */
  {0x7c, 0x7d}, /* KEY_YEN */
  {0x7d, 0xdb}, /* KEY_LEFTMETA */
  {0x7e, 0xdc}, /* KEY_RIGHTMETA */
  {0x7f, 0xdd}, /* KEY_COMPOSE */
  {0x80, 0xe8}, /* KEY_STOP */
  {0x81, 0x85}, /* KEY_AGAIN */
  {0x82, 0x86}, /* KEY_PROPS */
  {0x83, 0x87}, /* KEY_UNDO */
  {0x84, 0x8c}, /* KEY_FRONT */
  {0x85, 0xf8}, /* KEY_COPY */
  {0x86, 0x64}, /* KEY_OPEN */
  {0x87, 0x65}, /* KEY_PASTE */
  {0x88, 0xc1}, /* KEY_FIND */
  {0x89, 0xbc}, /* KEY_CUT */
  {0x8a, 0xf5}, /* KEY_HELP */
  {0x8b, 0x9e}, /* KEY_MENU */
  {0x8c, 0xa1}, /* KEY_CALC */
  {0x8d, 0x66}, /* KEY_SETUP */
  {0x8e, 0xdf}, /* KEY_SLEEP */
  {0x8f, 0xe3}, /* KEY_WAKEUP */
  {0x90, 0x67}, /* KEY_FILE */
  {0x91, 0x68}, /* KEY_SENDFILE */
  {0x92, 0x69}, /* KEY_DELETEFILE */
  {0x93, 0x93}, /* KEY_XFER */
  {0x94, 0x9f}, /* KEY_PROG1 */
  {0x95, 0x97}, /* KEY_PROG2 */
  {0x96, 0x82}, /* KEY_WWW */
  {0x97, 0x6a}, /* KEY_MSDOS */
  {0x98, 0x92}, /* KEY_SCREENLOCK */
  {0x99, 0x6b}, /* KEY_DIRECTION */
  {0x9a, 0xa6}, /* KEY_CYCLEWINDOWS */
  {0x9b, 0xec}, /* KEY_MAIL */
  {0x9c, 0xe6}, /* KEY_BOOKMARKS */
  {0x9d, 0xeb}, /* KEY_COMPUTER */
  {0x9e, 0xea}, /* KEY_BACK */
  {0x9f, 0xe9}, /* KEY_FORWARD */
  {0xa0, 0xa3}, /* KEY_CLOSECD */
  {0xa1, 0x6c}, /* KEY_EJECTCD */
  {0xa2, 0xfd}, /* KEY_EJECTCLOSECD */
  {0xa3, 0x99}, /* KEY_NEXTSONG */
  {0xa4, 0xa2}, /* KEY_PLAYPAUSE */
  {0xa5, 0x90}, /* KEY_PREVIOUSSONG */
  {0xa6, 0xa4}, /* KEY_STOPCD */
  {0xa7, 0xb1}, /* KEY_RECORD */
  {0xa8, 0x98}, /* KEY_REWIND */
  {0xa9, 0x63}, /* KEY_PHONE */
  {0xaa, 0x70}, /* KEY_ISO */
  {0xab, 0x81}, /* KEY_CONFIG */
  {0xac, 0xb2}, /* KEY_HOMEPAGE */
  {0xad, 0xe7}, /* KEY_REFRESH */
  {0xae, 0x71}, /* KEY_EXIT */
  {0xaf, 0x72}, /* KEY_MOVE */
  {0xb0, 0x88}, /* KEY_EDIT */
  {0xb1, 0x75}, /* KEY_SCROLLUP */
  {0xb2, 0x8f}, /* KEY_SCROLLDOWN */
  {0xb3, 0xf6}, /* KEY_KPLEFTPAREN */
  {0xb4, 0xfb}, /* KEY_KPRIGHTPAREN */
  {0xb5, 0x89}, /* KEY_NEW */
  {0xb6, 0x8a}, /* KEY_REDO */
  {0xb7, 0x5d}, /* KEY_F13 */
  {0xb8, 0x5e}, /* KEY_F14 */
  {0xb9, 0x5f}, /* KEY_F15 */
  {0xba, 0x55}, /* KEY_F16 */
  {0xbb, 0x83}, /* KEY_F17 */
  {0xbc, 0xf7}, /* KEY_F18 */
  {0xbd, 0x84}, /* KEY_F19 */
  {0xbe, 0x5a}, /* KEY_F20 */
  {0xbf, 0x74}, /* KEY_F21 */
  {0xc0, 0xf9}, /* KEY_F22 */
  {0xc1, 0x6d}, /* KEY_F23 */
  {0xc2, 0x6f}, /* KEY_F24 */
  {0xc3, 0x95}, /* unnamed */
  {0xc4, 0x96}, /* unnamed */
  {0xc5, 0x9a}, /* unnamed */
  {0xc6, 0x9b}, /* unnamed */
  {0xc7, 0xa7}, /* unnamed */
  {0xc8, 0xa8}, /* KEY_PLAYCD */
  {0xc9, 0xa9}, /* KEY_PAUSECD */
  {0xca, 0xab}, /* KEY_PROG3 */
  {0xcb, 0xac}, /* KEY_PROG4 */
  {0xcc, 0xad}, /* KEY_DASHBOARD */
  {0xcd, 0xa5}, /* KEY_SUSPEND */
  {0xce, 0xaf}, /* KEY_CLOSE */
  {0xcf, 0xb3}, /* KEY_PLAY */
  {0xd0, 0xb4}, /* KEY_FASTFORWARD */
  {0xd1, 0xb6}, /* KEY_BASSBOOST */
  {0xd2, 0xb9}, /* KEY_PRINT */
  {0xd3, 0xba}, /* KEY_HP */
  {0xd4, 0xbb}, /* KEY_CAMERA */
  {0xd5, 0xbd}, /* KEY_SOUND */
  {0xd6, 0xbe}, /* KEY_QUESTION */
  {0xd7, 0xbf}, /* KEY_EMAIL */
  {0xd8, 0xc0}, /* KEY_CHAT */
  {0xd9, 0xe5}, /* KEY_SEARCH */
  {0xda, 0xc2}, /* KEY_CONNECT */
  {0xdb, 0xc3}, /* KEY_FINANCE */
  {0xdc, 0xc4}, /* KEY_SPORT */
  {0xdd, 0xc5}, /* KEY_SHOP */
  {0xde, 0x94}, /* KEY_ALTERASE */
  {0xdf, 0xca}, /* KEY_CANCEL */
  {0xe0, 0xcc}, /* KEY_BRIGHTNESSDOWN */
  {0xe1, 0xd4}, /* KEY_BRIGHTNESSUP */
  {0xe2, 0xed}, /* KEY_MEDIA */
  {0xe3, 0xd6}, /* KEY_SWITCHVIDEOMODE */
  {0xe4, 0xd7}, /* KEY_KBDILLUMTOGGLE */
  {0xe5, 0xd8}, /* KEY_KBDILLUMDOWN */
  {0xe6, 0xd9}, /* KEY_KBDILLUMUP */
  {0xe7, 0xda}, /* KEY_SEND */
  {0xe8, 0xe4}, /* KEY_REPLY */
  {0xe9, 0x8e}, /* KEY_FORWARDMAIL */
  {0xea, 0xd5}, /* KEY_SAVE */
  {0xeb, 0xf0}, /* KEY_DOCUMENTS */
  {0xec, 0xf1}, /* KEY_BATTERY */
  {0xed, 0xf2}, /* KEY_BLUETOOTH */
  {0xee, 0xf3}, /* KEY_WLAN */
  {0xef, 0xf4}, /* KEY_UWB */
};

std::string get_ide_disk_target(int index) {
	char last_letter = 'a' + index;
	return std::string("hd") + last_letter;
}

std::string get_scsi_disk_target(int index) {
	char last_letter = 'a' + index;
	return std::string("sd") + last_letter;
}

const std::unordered_map<KeyboardButton, uint16_t> scancodes = {
	{KeyboardButton::ESC, 1},
	{KeyboardButton::ONE, 2},
	{KeyboardButton::TWO, 3},
	{KeyboardButton::THREE, 4},
	{KeyboardButton::FOUR, 5},
	{KeyboardButton::FIVE, 6},
	{KeyboardButton::SIX, 7},
	{KeyboardButton::SEVEN, 8},
	{KeyboardButton::EIGHT, 9},
	{KeyboardButton::NINE, 10},
	{KeyboardButton::ZERO, 11},
	{KeyboardButton::MINUS, 12},
	{KeyboardButton::EQUALSIGN, 13},
	{KeyboardButton::BACKSPACE, 14},
	{KeyboardButton::TAB, 15},
	{KeyboardButton::Q, 16},
	{KeyboardButton::W, 17},
	{KeyboardButton::E, 18},
	{KeyboardButton::R, 19},
	{KeyboardButton::T, 20},
	{KeyboardButton::Y, 21},
	{KeyboardButton::U, 22},
	{KeyboardButton::I, 23},
	{KeyboardButton::O, 24},
	{KeyboardButton::P, 25},
	{KeyboardButton::LEFTBRACE, 26},
	{KeyboardButton::RIGHTBRACE, 27},
	{KeyboardButton::ENTER, 28},
	{KeyboardButton::LEFTCTRL, 29},
	{KeyboardButton::A, 30},
	{KeyboardButton::S, 31},
	{KeyboardButton::D, 32},
	{KeyboardButton::F, 33},
	{KeyboardButton::G, 34},
	{KeyboardButton::H, 35},
	{KeyboardButton::J, 36},
	{KeyboardButton::K, 37},
	{KeyboardButton::L, 38},
	{KeyboardButton::SEMICOLON, 39},
	{KeyboardButton::APOSTROPHE, 40},
	{KeyboardButton::GRAVE, 41},
	{KeyboardButton::LEFTSHIFT, 42},
	{KeyboardButton::BACKSLASH, 43},
	{KeyboardButton::Z, 44},
	{KeyboardButton::X, 45},
	{KeyboardButton::C, 46},
	{KeyboardButton::V, 47},
	{KeyboardButton::B, 48},
	{KeyboardButton::N, 49},
	{KeyboardButton::M, 50},
	{KeyboardButton::COMMA, 51},
	{KeyboardButton::DOT, 52},
	{KeyboardButton::SLASH, 53},
	{KeyboardButton::RIGHTSHIFT, 54},
	{KeyboardButton::LEFTALT, 56},
	{KeyboardButton::SPACE, 57},
	{KeyboardButton::CAPSLOCK, 58},
	{KeyboardButton::F1, 59},
	{KeyboardButton::F2, 60},
	{KeyboardButton::F3, 61},
	{KeyboardButton::F4, 62},
	{KeyboardButton::F5, 63},
	{KeyboardButton::F6, 64},
	{KeyboardButton::F7, 65},
	{KeyboardButton::F8, 66},
	{KeyboardButton::F9, 67},
	{KeyboardButton::F10, 68},
	{KeyboardButton::F11, 87},
	{KeyboardButton::F12, 88},
	{KeyboardButton::NUMLOCK, 69},
	{KeyboardButton::KP_0, 82},
	{KeyboardButton::KP_1, 79},
	{KeyboardButton::KP_2, 80},
	{KeyboardButton::KP_3, 81},
	{KeyboardButton::KP_4, 75},
	{KeyboardButton::KP_5, 76},
	{KeyboardButton::KP_6, 77},
	{KeyboardButton::KP_7, 71},
	{KeyboardButton::KP_8, 72},
	{KeyboardButton::KP_9, 73},
	{KeyboardButton::KP_PLUS, 78},
	{KeyboardButton::KP_MINUS, 74},
	{KeyboardButton::KP_SLASH, 98},
	{KeyboardButton::KP_ASTERISK, 55},
	{KeyboardButton::KP_ENTER, 96},
	{KeyboardButton::KP_DOT, 83},
	{KeyboardButton::SCROLLLOCK, 70},
	{KeyboardButton::RIGHTCTRL, 97},
	{KeyboardButton::RIGHTALT, 100},
	{KeyboardButton::HOME, 102},
	{KeyboardButton::UP, 103},
	{KeyboardButton::PAGEUP, 104},
	{KeyboardButton::LEFT, 105},
	{KeyboardButton::RIGHT, 106},
	{KeyboardButton::END, 107},
	{KeyboardButton::DOWN, 108},
	{KeyboardButton::PAGEDOWN, 109},
	{KeyboardButton::INSERT, 110},
	{KeyboardButton::DELETE, 111},
	{KeyboardButton::LEFTMETA, 125},
	{KeyboardButton::RIGHTMETA, 126},
	{KeyboardButton::SCROLLUP, 177},
	{KeyboardButton::SCROLLDOWN, 178},
};

QemuVM::QemuVM(const nlohmann::json& config_): VM(config_),
	qemu_connect(vir::connect_open("qemu:///system"))
{

}

QemuVM::~QemuVM() {
	if (!is_defined()) {
		remove_disks();
	}
}

#ifdef __aarch64__
std::string QemuVM::compose_config() const {
	try {
		std::string string_config = fmt::format(R"(
<domain type='kvm'>
	<name>{}</name>
	<metadata>
		<testo:is_testo_related xmlns:testo='http://testo' value='true'/>
	</metadata>
	<memory unit='MiB'>{}</memory>
	<vcpu placement='static'>{}</vcpu>
		)", id(), config.at("ram").get<uint32_t>(), config.at("cpus").get<uint32_t>());

		string_config += R"(
	<os>
		<type arch='aarch64' machine='virt-5.0'>hvm</type>
		)";

		if (config.count("loader")) {
			string_config += fmt::format(R"(
		<loader readonly='yes' type='pflash'>{}</loader>
			)", config.at("loader").get<std::string>());
		} else {
			string_config += R"(
		<loader readonly='yes' type='pflash'>/usr/share/AAVMF/AAVMF_CODE.fd</loader>
			)";
		}

		string_config += fmt::format(R"(
		<nvram>{}</nvram>
		<bootmenu enable='yes'/>
	</os>
		)", current_nvram_path().string());

		string_config += fmt::format(R"(
	<features>
	</features>
	<cpu mode='host-passthrough'>
		<model fallback='forbid'/>
		<topology sockets='1' cores='{}' threads='1'/>
	</cpu>
	<clock offset='utc'/>
	<on_poweroff>destroy</on_poweroff>
	<on_reboot>restart</on_reboot>
	<on_crash>destroy</on_crash>
	<devices>
		<emulator>/usr/bin/qemu-system-aarch64</emulator>
		)", config.at("cpus").get<uint32_t>());

		size_t scsi_disk_counter = 0;

		if (config.count("disk")) {
			auto& disks = config.at("disk");
			for (auto& disk: disks) {
				std::string bus = disk.at("bus");
				if (bus == "ide") {
					throw std::runtime_error("IDE bus is not supported on ARM architecture");
				}
				std::string disk_name = disk.at("name");
				string_config += fmt::format(R"(
		<disk type='file' device='disk'>
			<driver name='qemu' type='qcow2'/>
			<source file='{}'/>
			<target dev='{}' bus='scsi'/>
			<alias name='ua-{}'/>
			<boot order='{}'/>
		</disk>
				)", disk_path(disk_name).generic_string(), get_scsi_disk_target(scsi_disk_counter), disk_name, 2 + scsi_disk_counter);
				++scsi_disk_counter;
			}
		}

		if (config.count("iso")) {
			string_config += fmt::format(R"(
		<disk type='file' device='cdrom'>
			<driver name='qemu' type='raw'/>
			<source file='{}'/>
			<target dev='{}' bus='scsi'/>
			<readonly/>
			<boot order='1'/>
		</disk>
			)", config.at("iso").get<std::string>(), get_scsi_disk_target(scsi_disk_counter));
		} else {
			string_config += fmt::format(R"(
		<disk type='file' device='cdrom'>
			<driver name='qemu' type='raw'/>
			<target dev='{}' bus='scsi'/>
			<readonly/>
		</disk>
			)", get_scsi_disk_target(scsi_disk_counter));
		}

		string_config += R"(

		<controller type='scsi' index='0' model='virtio-scsi'>
		</controller>

		<controller type='pci' index='0' model='pcie-root'/>
		<controller type='pci' index='1' model='pcie-root-port'>
			<model name='pcie-root-port'/>
			<target chassis='1' port='0x8'/>
		</controller>
		<controller type='pci' index='2' model='pcie-root-port'>
			<model name='pcie-root-port'/>
			<target chassis='2' port='0x9'/>
		</controller>
		<controller type='pci' index='3' model='pcie-root-port'>
			<model name='pcie-root-port'/>
			<target chassis='3' port='0xa'/>
		</controller>
		<controller type='pci' index='4' model='pcie-to-pci-bridge'>
			<model name='pcie-pci-bridge'/>
		</controller>
		<controller type='pci' index='5' model='pcie-root-port'>
			<model name='pcie-root-port'/>
			<target chassis='5' port='0xb'/>
		</controller>
		<controller type='pci' index='6' model='pcie-root-port'>
			<model name='pcie-root-port'/>
			<target chassis='6' port='0xc'/>
		</controller>
		<controller type='pci' index='7' model='pcie-root-port'>
			<model name='pcie-root-port'/>
			<target chassis='7' port='0xd'/>
		</controller>
		<controller type='pci' index='8' model='pcie-root-port'>
			<model name='pcie-root-port'/>
			<target chassis='8' port='0xe'/>
		</controller>
		<controller type='pci' index='9' model='pcie-root-port'>
			<model name='pcie-root-port'/>
			<target chassis='9' port='0xf'/>
		</controller>
		<controller type='pci' index='10' model='pcie-root-port'>
			<model name='pcie-root-port'/>
			<target chassis='10' port='0x10'/>
		</controller>
		)";

		string_config += R"(
		<controller type='virtio-serial' index='0'>
		</controller>
		<controller type='usb' index='0' model='ich9-ehci1'>
		</controller>
		<controller type='usb' index='0' model='ich9-uhci1'>
			<master startport='0'/>
		</controller>
		<controller type='usb' index='0' model='ich9-uhci2'>
			<master startport='2'/>
		</controller>
		<controller type='usb' index='0' model='ich9-uhci3'>
			<master startport='4'/>
		</controller>
		)";

		if (config.count("shared_folder")) {
			for (auto& shared_folder: config.at("shared_folder")) {
				string_config += fmt::format(R"(
		<filesystem type='mount' accessmode='mapped'>
			<driver type='path' wrpolicy='immediate'/>
			<source dir='{}'/>
			<target dir='{}'/>
				)", shared_folder.at("host_path").get<std::string>(),
					shared_folder.at("name").get<std::string>()
				);
				if (shared_folder.value("readonly", false)) {
					string_config += R"(
			<readonly/>
					)";
				}
				string_config += R"(
		</filesystem>
				)";
			}
		}

		string_config += R"(
		<serial type='pty'>
			<target type='system-serial' port='0'>
				<model name='pl011'/>
			</target>
		</serial>
		<console type='pty'>
			<target type='serial' port='0'/>
		</console>
		<channel type='unix'>
			<target type='virtio' name='negotiator.0'/>
		</channel>

		<input type='tablet' bus='usb'>
		</input>
		<input type='keyboard' bus='usb'/>
		<graphics type='spice' autoport='yes'>
			<listen type='address'/>
			<image compression='off'/>
			<gl enable='no'/>
		</graphics>
		<sound model='ich6'>
		</sound>
		)";

		string_config += R"(
		<video>
		)";

		if (config.count("video")) {
			auto videos = config.at("video");
			for (auto& video: videos) {
				auto video_model = video.value("adapter_type", video.value("qemu_mode", preferable_video_model(qemu_connect)));

				string_config += fmt::format(R"(
			<model type='{}' heads='1' primary='yes'/>
				)", video_model);
			}
		} else {
			string_config += fmt::format(R"(
			<model type='{}' heads='1' primary='yes'/>
			)", preferable_video_model(qemu_connect));
		}

		string_config += R"(
		</video>
		)";

		if (config.at("qemu_spice_agent")) {
			string_config += R"(
		<channel type='spicevmc'>
			<target type='virtio' name='com.redhat.spice.0'/>
		</channel>
			)";
		}

		string_config += R"(
		<redirdev bus='usb' type='spicevmc'>
		</redirdev>
		<redirdev bus='usb' type='spicevmc'>
		</redirdev>
	</devices>
</domain>
		)";

		return string_config;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Composing xml config")));
	}
}
#endif

#ifdef __x86_64__
std::string QemuVM::compose_config() const {
	try {
		std::string string_config = fmt::format(R"(
			<domain type='kvm'>
				<name>{}</name>
				<memory unit='MiB'>{}</memory>
				<vcpu placement='static'>{}</vcpu>
				<resource>
					<partition>/machine</partition>
				</resource>
				<features>
					<acpi/>
					<apic/>
					<vmport state='off'/>
				</features>
				<cpu mode='host-passthrough'>
					<model fallback='forbid'/>
					<topology sockets='1' cores='{}' threads='1'/>
				</cpu>
				<clock offset='utc'>
					<timer name='rtc' tickpolicy='catchup'/>
					<timer name='pit' tickpolicy='delay'/>
					<timer name='hpet' present='yes'/>
				</clock>
				<on_poweroff>destroy</on_poweroff>
				<on_reboot>restart</on_reboot>
				<on_crash>destroy</on_crash>
				<pm>
				</pm>
				<metadata>
					<testo:is_testo_related xmlns:testo='http://testo' value='true'/>
				</metadata>
		)", id(), config.at("ram").get<uint32_t>(), config.at("cpus").get<uint32_t>(), config.at("cpus").get<uint32_t>());

		string_config += R"(
			<os>
				<type>hvm</type>
				<boot dev='cdrom'/>
				<boot dev='hd'/>
		)";

		if (config.count("loader")) {
			string_config += fmt::format(R"(
				<loader readonly='yes' type='rom'>{}</loader>
			)", config.at("loader").get<std::string>());
		}

		string_config += R"(
				<bootmenu enable='yes' timeout='1000'/>
			</os>
		)";

		string_config += R"(
			<devices>
				<controller type='ide' index='0'>
				</controller>
				<controller type='scsi' index='0' model='virtio-scsi'>
				</controller>
				<controller type='virtio-serial' index='0'>
				</controller>
				<controller type='pci' index='0' model='pci-root'/>
				<serial type='pty'>
					<target type='isa-serial' port='0'>
						<model name='isa-serial'/>
					</target>
				</serial>
				<console type='pty'>
					<target type='serial' port='0'/>
				</console>
				<channel type='unix'>
					<target type='virtio' name='negotiator.0'/>
				</channel>
				<input type='tablet' bus='usb'>
				</input>
				<input type='mouse' bus='ps2'/>
				<input type='keyboard' bus='ps2'/>
				<graphics type='spice' autoport='yes'>
					<listen type='address'/>
					<image compression='off'/>
				</graphics>
				<sound model='ich6'>
				</sound>
				<redirdev bus='usb' type='spicevmc'>
				</redirdev>
				<redirdev bus='usb' type='spicevmc'>
				</redirdev>
				<memballoon model='virtio'>
				</memballoon>
		)";

		if (config.count("shared_folder")) {
			for (auto& shared_folder: config.at("shared_folder")) {
				string_config += fmt::format(R"(
					<filesystem type='mount' accessmode='mapped'>
						<driver type='path' wrpolicy='immediate'/>
						<source dir='{}'/>
						<target dir='{}'/>
				)", shared_folder.at("host_path").get<std::string>(),
					shared_folder.at("name").get<std::string>()
				);
				if (shared_folder.value("readonly", false)) {
					string_config += R"(
						<readonly/>
					)";
				}
				string_config += R"(
					</filesystem>
				)";
			}
		}

		if (config.at("qemu_enable_usb3")) {
			string_config += R"(
				<controller type='usb' index='0' model='nec-xhci' ports='15'>
				</controller>
			)";
		} else {
			string_config += R"(
				<controller type='usb' index='0' model='ich9-ehci1'>
				</controller>
				<controller type='usb' index='0' model='ich9-uhci1'>
				</controller>
				<controller type='usb' index='0' model='ich9-uhci2'>
				</controller>
				<controller type='usb' index='0' model='ich9-uhci3'>
				</controller>
			)";
		}

		string_config += R"(
			<video>
		)";

		if (config.count("video")) {
			auto videos = config.at("video");
			for (auto& video: videos) {
				auto video_model = video.value("adapter_type", video.value("qemu_mode", preferable_video_model(qemu_connect)));

				string_config += fmt::format(R"(
					<model type='{}' heads='1' primary='yes'/>
				)", video_model);
			}
		} else {
			string_config += fmt::format(R"(
				<model type='{}' heads='1' primary='yes'/>
			)", preferable_video_model(qemu_connect));
		}

		string_config += R"(
			</video>
		)";

		size_t ide_disk_counter = 0;
		size_t scsi_disk_counter = 0;

		if (config.count("disk")) {
			auto& disks = config.at("disk");
			for (auto& disk: disks) {
				std::string bus = disk.at("bus");
				std::string target;
				if (bus == "ide") {
					target = get_ide_disk_target(ide_disk_counter);
					++ide_disk_counter;
				} else {
					target = get_scsi_disk_target(scsi_disk_counter);
					++scsi_disk_counter;
				}
				std::string disk_name = disk.at("name");
				string_config += fmt::format(R"(
					<disk type='file' device='disk'>
						<driver name='qemu' type='qcow2'/>
						<source file='{}'/>
						<target dev='{}' bus='{}'/>
						<alias name='ua-{}'/>
					</disk>
				)", disk_path(disk_name).generic_string(), target, bus, disk_name);
			}
		}

		if (config.count("iso")) {
			string_config += fmt::format(R"(
				<disk type='file' device='cdrom'>
					<driver name='qemu' type='raw'/>
					<source file='{}'/>
					<target dev='{}' bus='ide'/>
					<readonly/>
				</disk>
			)", config.at("iso").get<std::string>(), get_ide_disk_target(ide_disk_counter));
		} else {
			string_config += fmt::format(R"(
				<disk type='file' device='cdrom'>
					<driver name='qemu' type='raw'/>
					<target dev='{}' bus='ide'/>
					<readonly/>
				</disk>
			)", get_ide_disk_target(ide_disk_counter));
		}

		if (config.at("qemu_spice_agent")) {
			string_config += R"(
			<channel type='spicevmc'>
				<target type='virtio' name='com.redhat.spice.0'/>
			</channel>)";
		}

		string_config += "\n </devices> \n </domain>";

		return string_config;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Composing xml config")));
	}
}
#endif

void QemuVM::install() {
	try {
		create_disks();

		if (!config.count("qemu_enable_usb3")) {
			config["qemu_enable_usb3"] = false;
		}

		if (!config.count("qemu_spice_agent")) {
			config["qemu_spice_agent"] = false;
		}

		std::string string_config = compose_config();

		pugi::xml_document xml_config;
		xml_config.load_string(string_config.c_str());
		qemu_connect.domain_define_xml(xml_config);

		if (config.count("nic")) {
			auto nics = config.at("nic");
			for (auto& nic: nics) {
				auto plugged = nic.value("plugged", true);
				plug_nic(nic.at("name").get<std::string>());
				if (!plugged) {
					unplug_nic(nic.at("name").get<std::string>());
				}
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Performing install")));
	}
}

void QemuVM::undefine() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		if (use_external_snapshots()) {
			remove_disks();
		} else {
			for (auto& snapshot: domain.snapshots({VIR_DOMAIN_SNAPSHOT_LIST_ROOTS})) {
				snapshot.destroy({VIR_DOMAIN_SNAPSHOT_DELETE_CHILDREN});
			}
		}
		domain.undefine();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Undefining vm {}", id())));
	}
}

nlohmann::json QemuVM::make_snapshot(const std::string& snapshot) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		nlohmann::json umounted_folders = nlohmann::json::array();
		if (config.count("shared_folder") && config.at("shared_folder").size() && (state() == VmState::Suspended)) {
			resume();
			QemuGuestAdditions ga(domain);
			if (ga.is_avaliable(1500ms)) {
				for (auto& shared_folder: config.at("shared_folder")) {
					auto folder_status = ga.get_shared_folder_status(shared_folder.at("name"));
					if (folder_status.at("is_mounted")) {
						ga.umount(folder_status.at("name"), false);
						umounted_folders.push_back(folder_status);
					}
				}
			}
			suspend();
		}

		if (use_external_snapshots()) {
			bool is_paused = domain.state() == VIR_DOMAIN_PAUSED;

			if (is_paused) {
				fs::create_directories(memory_snapshot_path(snapshot).parent_path());
				domain.save(memory_snapshot_path(snapshot));
			}

			if (fs::exists(current_nvram_path())) {
				fs_copy(current_nvram_path(), nvram_snapshot_path(snapshot));
			} else {
				throw std::runtime_error("File " + current_nvram_path().string() + " does not exist");
			}


			if (config.count("disk")) {
				auto& disks = config.at("disk");
				for (size_t i = 0; i < disks.size(); ++i) {
					auto& disk = disks[i];
					std::string disk_name = disk.at("name");
					fs::path path = disk_path(disk_name);
					fs::path snapshot_path = disk_snapshot_path(disk_name, snapshot);
					fs::rename(path, snapshot_path);
					create_img_with_backing_file(path, snapshot_path);
				}
			}

			if (is_paused) {
				qemu_connect.restore_domain(memory_snapshot_path(snapshot));
				domain = qemu_connect.domain_lookup_by_name(id());
			}
		} else {
			pugi::xml_document xml_config;
			xml_config.load_string(fmt::format(R"(
				<domainsnapshot>
					<name>{}</name>
				</domainsnapshot>
				)", snapshot).c_str());

			auto snap = domain.snapshot_create_xml(xml_config);

			// If we created the _init snapshot
			if (domain.snapshots().size() == 1) {
				snap.destroy();
				snap = domain.snapshot_create_xml(xml_config);
			}
		}

		if (umounted_folders.size()) {
			resume();
			QemuGuestAdditions ga(domain);
			for (auto& folder_status: umounted_folders) {
				ga.mount(folder_status.at("name"), folder_status.at("guest_path").get<std::string>(), false);
			}
			suspend();
		}

		auto result = nlohmann::json::object();
		result["config"] = domain.dump_xml_base64();
		result["nics"] = nic_pci_map;
		result["automaticaly_umounted_shared_folders"] = umounted_folders;
		return result;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Taking snapshot {}", snapshot)));
	}

}

void QemuVM::rollback(const std::string& snapshot, const nlohmann::json& opaque) {
	try {
		{
			auto domain = qemu_connect.domain_lookup_by_name(id());
			if (domain.state() != VIR_DOMAIN_SHUTOFF) {
				domain.stop();
			}
		}

		nic_pci_map.clear();

		auto& nics = opaque.at("nics");
		for (auto it = nics.begin(); it != nics.end(); ++it) {
			nic_pci_map[it.key()] = it.value().get<std::string>();
		}

		auto domain = qemu_connect.domain_define_xml_base64(opaque.at("config"));

		if (use_external_snapshots()) {
			if (fs::exists(nvram_snapshot_path(snapshot))) {
				fs_copy(nvram_snapshot_path(snapshot), current_nvram_path());
			} else {
				throw std::runtime_error("File " + nvram_snapshot_path(snapshot).string() + " does not exist");
			}

			if (config.count("disk")) {
				auto& disks = config.at("disk");
				for (size_t i = 0; i < disks.size(); ++i) {
					auto& disk = disks[i];
					std::string disk_name = disk.at("name");
					fs::path path = disk_path(disk_name);
					fs::path snapshot_path = disk_snapshot_path(disk_name, snapshot);
					if (fs::exists(path)) {
						fs::remove(path);
					}
					create_img_with_backing_file(path, snapshot_path);
				}
			}

			if (fs::exists(memory_snapshot_path(snapshot))) {
				qemu_connect.restore_domain(memory_snapshot_path(snapshot));
			}
		} else {
			auto snap = domain.snapshot_lookup_by_name(snapshot);
			domain.revert_to_snapshot(snap);
		}

		if (opaque.count("automaticaly_umounted_shared_folders") && opaque.at("automaticaly_umounted_shared_folders").size() && (state() == VmState::Suspended)) {
			resume();
			QemuGuestAdditions ga(domain);
			for (auto& folder_status: opaque.at("automaticaly_umounted_shared_folders")) {
				ga.mount(folder_status.at("name"), folder_status.at("guest_path").get<std::string>(), false);
			}
			suspend();
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Performing rollback error"));
	}
}

void QemuVM::hold(KeyboardButton button) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		uint32_t scancode = virKeyCodeTable_rfb.at(scancodes.at(button));

		std::string json_command = fmt::format(R"(
			{{
				"execute": "input-send-event",
				"arguments": {{
					"events": [
						{{
							"type": "key",
							"data": {{
								"down": true,
								"key": {{
									"type": "number",
									"data": {}
								}}
							}}
						}}
					]
				}}
			}}
		)", scancode);

		auto result = domain.monitor_command(json_command);

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Holding button error"));
	}
}


void QemuVM::release(KeyboardButton button) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		uint32_t scancode = virKeyCodeTable_rfb.at(scancodes.at(button));

		std::string json_command = fmt::format(R"(
			{{
				"execute": "input-send-event",
				"arguments": {{
					"events": [
						{{
							"type": "key",
							"data": {{
								"down": false,
								"key": {{
									"type": "number",
									"data": {}
								}}
							}}
						}}
					]
				}}
			}}
		)", scancode);

		auto result = domain.monitor_command(json_command);

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Releasing button error"));
	}
}

void QemuVM::mouse_move_abs(uint32_t x, uint32_t y) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto tmp_screen = screenshot();

		double x_pos = double(32768) / double(tmp_screen.w) * double(x);
		double y_pos = double(32768) / double(tmp_screen.h) * double(y);

		if ((int)x_pos == 0) {
			x_pos = 1;
		}

		if ((int)y_pos == 0) {
			y_pos = 1;
		}

		std::string json_command = fmt::format(R"(
			{{
				"execute": "input-send-event",
				"arguments": {{
					"events": [
						{{
							"type": "abs",
							"data": {{
								"axis": "x",
								"value": {}
							}}

						}},
						{{
							"type": "abs",
							"data": {{
								"axis": "y",
								"value": {}
							}}
						}}
					]
				}}
			}}
		)", (int)x_pos, (int)y_pos);

		auto result = domain.monitor_command(json_command);

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Mouse move error"));
	}
}

void QemuVM::mouse_move_rel(int x, int y) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		std::string json_command = fmt::format(R"(
			{{
				"execute": "input-send-event",
				"arguments": {{
					"events": [
						{{
							"type": "rel",
							"data": {{
								"axis": "x",
								"value": {}
							}}
						}},
						{{
							"type": "rel",
							"data": {{
								"axis": "y",
								"value": {}
							}}
						}}
					]
				}}
			}}
		)", x, y);

		auto result = domain.monitor_command(json_command);

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Mouse move error"));
	}
}

void QemuVM::mouse_hold(MouseButton button) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		std::string json_command = fmt::format(R"(
			{{
				"execute": "input-send-event",
				"arguments": {{
					"events": [
						{{
							"type": "btn",
							"data": {{
								"down": true,
								"button": "{}"
							}}
						}}
					]
				}}
			}}
		)", mouse_button_to_str(button));

		auto result = domain.monitor_command(json_command);

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Mouse press button error"));
	}
}


void QemuVM::mouse_release(MouseButton button) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		std::string json_command = fmt::format(R"(
			{{
				"execute": "input-send-event",
				"arguments": {{
					"events": [
						{{
							"type": "btn",
							"data": {{
								"down": false,
								"button": "{}"
							}}
						}}
					]
				}}
			}}
		)", mouse_button_to_str(button));

		auto result = domain.monitor_command(json_command);

		if (result.count("error")) {
			throw std::runtime_error(result.at("error").at("desc").get<std::string>());
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Mouse release button error"));
	}
}

bool QemuVM::is_nic_plugged(const std::string& nic) const {
	try {
		auto config = qemu_connect.domain_lookup_by_name(id()).dump_xml();
		auto devices = config.first_child().child("devices");
		std::string pci_addr = nic_pci_map.at(nic);

		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			auto type = std::string(nic_node.attribute("type").value());
			if (type != "network" && type != "direct") {
				continue;
			}

			std::string pci_address;
			pci_address += std::string(nic_node.child("address").attribute("bus").value()).substr(2) + ":";
			pci_address += std::string(nic_node.child("address").attribute("slot").value()).substr(2) + ".";
			pci_address += std::string(nic_node.child("address").attribute("function").value()).substr(2);

			if (pci_address == pci_addr) {
				return true;
			}
		}
		return false;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking if nic {} is plugged", nic)));
	}
}

std::set<std::string> QemuVM::plugged_nics() const {
	auto domain = qemu_connect.domain_lookup_by_name(id());
	auto xml_config = domain.dump_xml();

	auto devices = xml_config.first_child().child("devices");

	std::set<std::string> result;

	for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
		std::string type = std::string(nic_node.attribute("type").value());

		if (type != "network" && type != "direct") {
			continue;
		}

		std::string pci_address;
		pci_address += std::string(nic_node.child("address").attribute("bus").value()).substr(2) + ":";
		pci_address += std::string(nic_node.child("address").attribute("slot").value()).substr(2) + ".";
		pci_address += std::string(nic_node.child("address").attribute("function").value()).substr(2);

		result.insert(pci_address);
	}

	return result;
}

void QemuVM::plug_nic(const std::string& nic) {
	try {
		std::string string_config;

		for (auto& nic_json: config.at("nic")) {
			if (nic_json.at("name") == nic) {
				if (nic_json.count("attached_to")) {
					std::string source_network = config.at("prefix").get<std::string>();
					source_network += nic_json.at("attached_to").get<std::string>();

					string_config += fmt::format(R"(
						<interface type='network'>
							<source network='{}'/>
					)", source_network);
				} else if (nic_json.count("attached_to_dev")) {
					std::string dev = nic_json.at("attached_to_dev").get<std::string>();
					string_config += fmt::format(R"(
						<interface type='direct'>
							<source dev='{}' mode='bridge'/>
					)", dev);
				} else {
					throw std::runtime_error("Should never happen");
				}

#ifdef __aarch64__
				string_config += "<rom enabled='no'/>";
#endif

				if (nic_json.count("mac")) {
					string_config += fmt::format("\n<mac address='{}'/>", nic_json.at("mac").get<std::string>());
				}

				if (nic_json.count("adapter_type")) {
					string_config += fmt::format("\n<model type='{}'/>", nic_json.at("adapter_type").get<std::string>());
				}

				//libvirt suggests that everything you do in aliases must be prefixed with "ua-nic-"
				std::string nic_name = std::string("ua-nic-");
				nic_name += nic_json.at("name").get<std::string>();
				string_config += fmt::format("\n<link state='up'/>");
				string_config += fmt::format("\n<alias name='{}'/>", nic_name);
				string_config += fmt::format("\n</interface>");

				break;
			}
		}
		auto domain = qemu_connect.domain_lookup_by_name(id());

		//TODO: check if CURRENT is enough
		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		pugi::xml_document nic_config;
		nic_config.load_string(string_config.c_str());

		auto already_plugged_nics = plugged_nics();
		domain.attach_device(nic_config, flags);

		auto new_plugged_nics = plugged_nics();

		std::set<std::string> diff;

		std::set_difference(new_plugged_nics.begin(), new_plugged_nics.end(), already_plugged_nics.begin(), already_plugged_nics.end(),
			std::inserter(diff, diff.begin()));

		std::string pci_addr = *diff.begin();
		nic_pci_map[nic] = pci_addr;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Plugging nic {}", nic)));
	}
}

void QemuVM::unplug_nic(const std::string& nic) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");
		std::string pci_addr = nic_pci_map.at(nic);

		//TODO: check if CURRENT is enough
		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			auto type = std::string(nic_node.attribute("type").value());
			if (type != "network" && type != "direct") {
				continue;
			}

			std::string pci_address;
			pci_address += std::string(nic_node.child("address").attribute("bus").value()).substr(2) + ":";
			pci_address += std::string(nic_node.child("address").attribute("slot").value()).substr(2) + ".";
			pci_address += std::string(nic_node.child("address").attribute("function").value()).substr(2);

			if (pci_address == pci_addr) {
				domain.detach_device(nic_node, flags);
				nic_pci_map[nic] = "";
				return;
			}
		}

		throw std::runtime_error("Nic with address " + pci_addr + " not found");
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Unplugging nic {}", nic)));
	}
}

bool QemuVM::is_link_plugged(const std::string& nic) const {
	try {
		auto config = qemu_connect.domain_lookup_by_name(id()).dump_xml();
		auto devices = config.first_child().child("devices");
		std::string pci_addr = nic_pci_map.at(nic);
		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			auto type = std::string(nic_node.attribute("type").value());
			if (type != "network" && type != "direct") {
				continue;
			}

			std::string pci_address;
			pci_address += std::string(nic_node.child("address").attribute("bus").value()).substr(2) + ":";
			pci_address += std::string(nic_node.child("address").attribute("slot").value()).substr(2) + ".";
			pci_address += std::string(nic_node.child("address").attribute("function").value()).substr(2);

			if (pci_address == pci_addr) {
				if (nic_node.child("link").empty()) {
					return false;
				}

				std::string state = nic_node.child("link").attribute("state").value();

				if (state == "up") {
					return true;
				} else if (state == "down") {
					return false;
				}
			}
		}
		throw std::runtime_error("Nic with address " + pci_addr + " not found");
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking link status on nic {}", nic)));
	}
}

void QemuVM::set_link(const std::string& nic, bool is_connected) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");
		std::string pci_addr = nic_pci_map.at(nic);
		for (auto nic_node = devices.child("interface"); nic_node; nic_node = nic_node.next_sibling("interface")) {
			auto type = std::string(nic_node.attribute("type").value());
			if (type != "network" && type != "direct") {
				continue;
			}

			std::string pci_address;
			pci_address += std::string(nic_node.child("address").attribute("bus").value()).substr(2) + ":";
			pci_address += std::string(nic_node.child("address").attribute("slot").value()).substr(2) + ".";
			pci_address += std::string(nic_node.child("address").attribute("function").value()).substr(2);

			if (pci_address == pci_addr) {
				if (is_connected) { //connect link
					//if we have set link attribute - just change state to up
					if (!nic_node.child("link").empty()) {
						nic_node.child("link").attribute("state").set_value("up");
					}
				} else { //disconnect link
					//if we have set link attribute - set it to down
					if (!nic_node.child("link").empty()) {
						nic_node.child("link").attribute("state").set_value("down");
					} else {
						auto link = nic_node.insert_child_before("link", nic_node.child("alias"));
						link.append_attribute("state") = "down";
					}
				}

				std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

				if (domain.is_active()) {
					flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
				}

				domain.update_device(nic_node, flags);
				return;
			}
		}
		throw std::runtime_error("Nic with address " + pci_addr + " not found");
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Setting link status on nic {}", nic)));
	}
}

bool QemuVM::is_flash_plugged(std::shared_ptr<FlashDrive> fd) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		std::string result = "";

		for (auto disk = devices.child("disk"); disk; disk = disk.next_sibling("disk")) {
			if (std::string(disk.attribute("device").value()) != "disk") {
				continue;
			}

			if (std::string(disk.child("target").attribute("bus").value()) == "usb") {
				result = disk.child("source").attribute("file").value();

				//nullptr fd means "Any" flash drive
				if (!fd) {
					return true;
				}

				if (result == fd->img_path().generic_string()) {
					return true;
				}
			}
		}

		return false;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking if flash drive {} is plugged", fd->name())));
	}
}

bool QemuVM::is_target_free(const std::string& target) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		for (auto disk = devices.child("disk"); disk; disk = disk.next_sibling("disk"))
		{
			if (std::string(disk.child("target").attribute("dev").value()) == target) {
				return false;
			}
		}

		return true;
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("is_target_free for target {}", target)));
	}
}

void QemuVM::attach_flash_drive(const std::string& img_path) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		int free_target_index = -1;

		for (size_t i = 0; i < 20; i++) {
			if (is_target_free(get_scsi_disk_target(i))) {
				free_target_index = i;
				break;
			}
		}

		if (free_target_index < 0) {
			throw std::runtime_error("Can't attach flash drive - no free slots");
		}

		std::string string_config = fmt::format(R"(
			<disk type='file'>
				<driver name='qemu' type='qcow2'/>
				<source file='{}'/>
				<target dev='{}' bus='usb' removable='on'/>
			</disk>
			)", img_path, get_scsi_disk_target(free_target_index));

		//we just need to create new device
		//TODO: check if CURRENT is enough
		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CONFIG, VIR_DOMAIN_DEVICE_MODIFY_CURRENT};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		pugi::xml_document disk_config;
		disk_config.load_string(string_config.c_str());

		domain.attach_device(disk_config, flags);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Attaching flash drive {}", img_path)));
	}

}

void QemuVM::plug_flash_drive(std::shared_ptr<FlashDrive> fd) {
	try {
		attach_flash_drive(fd->img_path());
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Plugging flash drive {}", fd->name())));
	}
}

bool QemuVM::is_hostdev_plugged() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		for (auto hostdev = devices.child("hostdev"); hostdev; hostdev = hostdev.next_sibling("hostdev")) {
			return true;
		}

		return false;

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking for plugged hostdevs")));
	}
}

void QemuVM::plug_hostdev_usb(const std::string& addr) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());

		auto parsed_addr = parse_usb_addr(addr);

		std::string string_config = fmt::format(R"(
			<hostdev mode='subsystem' type='usb'>
				<source>
					<address bus='{:#x}' device='{:#x}'/>
				</source>
			  </hostdev>
			)", parsed_addr.first, parsed_addr.second);

		//we just need to create new device
		//TODO: check if CURRENT is enough
		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CONFIG, VIR_DOMAIN_DEVICE_MODIFY_CURRENT};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		pugi::xml_document disk_config;
		disk_config.load_string(string_config.c_str());

		domain.attach_device(disk_config, flags);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Plugging host dev usb device {}", addr)));
	}
}

void QemuVM::unplug_hostdev_usb(const std::string& addr) {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		//TODO: check if CURRENT is enough
		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		auto parsed_addr = parse_usb_addr(addr);

		bool found = false;

		for (auto hostdev = devices.child("hostdev"); hostdev; hostdev = hostdev.next_sibling("hostdev")) {
			auto hostdev_addr = hostdev.child("source").child("address");

			int bus_id = std::stoi(hostdev_addr.attribute("bus").value(), 0, 0);
			int dev_id = std::stoi(hostdev_addr.attribute("device").value(), 0, 0);

			if ((bus_id == parsed_addr.first)
				&& (dev_id == parsed_addr.second))
			{
				domain.detach_device(hostdev, flags);
				found = true;
				break;
			}
		}

		if (!found) {
			throw std::runtime_error("Requested usb device is not plugged into the virtual machine");
		}

	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Unplugging host dev usb device {}", addr)));
	}
}

void QemuVM::detach_flash_drive() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto devices = config.first_child().child("devices");

		//TODO: check if CURRENT is enough
		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		for (auto disk = devices.child("disk"); disk; disk = disk.next_sibling("disk")) {
			if (std::string(disk.attribute("device").value()) != "disk") {
				continue;
			}

			if (std::string(disk.child("target").attribute("bus").value()) == "usb") {
				domain.detach_device(disk, flags);
				break;
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Detaching flash drive"));
	}
}

//for now it's just only one flash drive possible
void QemuVM::unplug_flash_drive(std::shared_ptr<FlashDrive> fd) {
	try {
		detach_flash_drive();
	} catch (const std::string& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Unplugging flash drive {}", fd->name())));
	}
}


bool QemuVM::is_dvd_plugged() const {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto cdrom = config.first_child().child("devices").find_child_by_attribute("device", "cdrom");
		if (cdrom.child("source").empty()) {
			return false;
		}
		return !cdrom.child("source").attribute("file").empty();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Checking if dvd is plugged"));
	}
}

void QemuVM::plug_dvd(fs::path path) {
	try {
		if (!fs::exists(path)) {
			throw std::runtime_error("specified iso file does not exist: " + path.generic_string());
		}

		if (!fs::is_regular_file(path)) {
			throw std::runtime_error("specified iso is not a regular file: " + path.generic_string());
		}
		auto domain = qemu_connect.domain_lookup_by_name(id());

		if (is_dvd_plugged()) {
			throw std::runtime_error("Some dvd is already plugged in");
		}

		auto config = domain.dump_xml();
		auto cdrom = config.first_child().child("devices").find_child_by_attribute("device", "cdrom");
		cdrom.remove_child("source");
		cdrom.append_child("source").append_attribute("file").set_value(path.generic_string().c_str());

		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CONFIG, VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_FORCE};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		domain.update_device(cdrom, flags);
	} catch (const std::string& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("plugging dvd {}", path.generic_string())));
	}
}

void QemuVM::unplug_dvd() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto config = domain.dump_xml();
		auto cdrom = config.first_child().child("devices").find_child_by_attribute("device", "cdrom");

		if (!is_dvd_plugged()) {
			throw std::runtime_error("Dvd is already unplugged");
		}

		cdrom.remove_child("source");

		std::vector<virDomainDeviceModifyFlags> flags = {VIR_DOMAIN_DEVICE_MODIFY_CURRENT, VIR_DOMAIN_DEVICE_MODIFY_CONFIG, VIR_DOMAIN_DEVICE_MODIFY_FORCE};

		if (domain.is_active()) {
			flags.push_back(VIR_DOMAIN_DEVICE_MODIFY_LIVE);
		}

		domain.update_device(cdrom, flags);
	} catch (const std::string& error) {
		std::throw_with_nested(std::runtime_error("Unplugging dvd"));
	}

}

void QemuVM::start() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto xml = domain.dump_xml();
		xml.first_child().child("cpu");
		pugi::xml_document cpu;
		cpu.load_string(fmt::format(R"(
			<cpu mode='host-passthrough'>
				<model fallback='forbid'/>
				<topology sockets='1' cores='{}' threads='1'/>
			</cpu>
		)", config.at("cpus").get<uint32_t>()).c_str());
		xml.first_child().append_copy(cpu.first_child());
		qemu_connect.domain_define_xml(xml);
		domain = qemu_connect.domain_lookup_by_name(id());
		domain.start();
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Starting vm"));
	}
}

void QemuVM::stop() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		domain.stop();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Stopping vm"));
	}
}

void QemuVM::power_button() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		domain.shutdown();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Shutdowning vm"));
	}
}

void QemuVM::suspend() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		domain.suspend();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Suspending vm"));
	}
}

void QemuVM::resume() {
	try {
		coro::Timeout timeout(10s);
		coro::Timer timer;
		while (true) {
			{
				auto domain = qemu_connect.domain_lookup_by_name(id());
				domain.resume();
			}
			{
				auto domain = qemu_connect.domain_lookup_by_name(id());
				if (domain.state() == VIR_DOMAIN_RUNNING) {
					return;
				} else {
					timer.waitFor(100ms);
				}
			}
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Resuming vm"));
	}
}

stb::Image<stb::RGB> QemuVM::screenshot() {
	auto domain = qemu_connect.domain_lookup_by_name(id());

	if (domain.state() != VIR_DOMAIN_RUNNING) {
		return {};
	}

	auto stream = qemu_connect.new_stream();
	auto mime = domain.screenshot(stream);

	if (!screenshot_buffer.size()) {
		screenshot_buffer.resize(10'000'000);
	}

	size_t bytes = stream.recv_all(screenshot_buffer.data(), screenshot_buffer.size());

	stream.finish();

	stb::Image<stb::RGB> screenshot(screenshot_buffer.data(), bytes);
	return screenshot;
}

bool QemuVM::has_snapshot(const std::string& snapshot) {
	try {
		if (use_external_snapshots()) {
			if (!fs::exists(nvram_snapshot_path(snapshot))) {
				return false;
			}

			if (config.count("disk")) {
				auto& disks = config.at("disk");
				for (size_t i = 0; i < disks.size(); ++i) {
					auto& disk = disks[i];
					std::string disk_name = disk.at("name");
					fs::path snapshot_path = disk_snapshot_path(disk_name, snapshot);
					if (!fs::exists(snapshot_path)) {
						return false;
					}
				}
			}

			return true;
		} else {
			auto domain = qemu_connect.domain_lookup_by_name(id());
			auto snapshots = domain.snapshots();
			for (auto& snap: snapshots) {
				if (snap.name() == snapshot) {
					return true;
				}
			}
			return false;
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking whether vm has snapshot {}", snapshot)));
	}
}

void QemuVM::delete_snapshot(const std::string& snapshot) {
	try {
		if (use_external_snapshots()) {
			if (fs::exists(nvram_snapshot_path(snapshot))) {
				fs::remove(nvram_snapshot_path(snapshot));
			}

			if (fs::exists(memory_snapshot_path(snapshot))) {
				fs::remove(memory_snapshot_path(snapshot));
			}

			if (config.count("disk")) {
				auto& disks = config.at("disk");
				for (size_t i = 0; i < disks.size(); ++i) {
					auto& disk = disks[i];
					std::string disk_name = disk.at("name");
					fs::path snapshot_path = disk_snapshot_path(disk_name, snapshot);
					if (fs::exists(snapshot_path)) {
						fs::remove(snapshot_path);
					}
				}
			}
		} else {
			auto domain = qemu_connect.domain_lookup_by_name(id());
			auto vir_snapshot = domain.snapshot_lookup_by_name(snapshot);
			vir_snapshot.destroy();
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Deleting snapshot with children"));
	}
}

bool QemuVM::is_defined() const {
	auto domains = qemu_connect.domains({VIR_CONNECT_LIST_DOMAINS_PERSISTENT});
	for (auto& domain: domains) {
		if (domain.name() == id()) {
			return true;
		}
	}
	return false;
}

VmState QemuVM::state() const {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		auto state = domain.state();
		if (state == VIR_DOMAIN_SHUTOFF) {
			return VmState::Stopped;
		} else if (state == VIR_DOMAIN_RUNNING) {
			return VmState::Running;
		} else if (state == VIR_DOMAIN_PAUSED) {
			return VmState::Suspended;
		} else {
			return VmState::Other;
		}
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Checking whether vm is running"));
	}
}

std::shared_ptr<GuestAdditions> QemuVM::guest_additions() {
	try {
		auto domain = qemu_connect.domain_lookup_by_name(id());
		return std::make_shared<QemuGuestAdditions>(domain);
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Connecting to guest additions channel"));
	}
}

bool QemuVM::is_my_disk(const fs::path& path) const {
	std::string disk_name = path.filename();
	if (disk_name.find("@") == std::string::npos) {
		return false;
	}
	std::string prefix = disk_name.substr(0, disk_name.find("@"));
	return prefix == id();
}

void QemuVM::remove_disks() {
	try {
		if (use_external_snapshots()) {
			for (fs::path dir: {disks_dir(), memory_dir(), nvram_dir()}) {
				if (!fs::exists(dir)) {
					continue;
				}
				for (auto& entry: fs::directory_iterator{dir}) {
					if (is_my_disk(entry)) {
						fs::remove(entry);
					}
				}
			}
		} else {
			auto pool = qemu_connect.storage_pool_lookup_by_name("testo-storage-pool");

			//TODO

			for (auto& vol: pool.volumes()) {
				if (is_my_disk(vol.name())) {
					vol.erase();
				}
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Removing existing disks"));
	}
}

void QemuVM::create_new_disk(const std::string& name, uint32_t size) {
	if (use_external_snapshots()) {
		create_empty_img(disk_path(name), size);
	} else {
		pugi::xml_document xml_config;
		xml_config.load_string(fmt::format(R"(
			<volume type='file'>
				<name>{}</name>
				<source>
				</source>
				<capacity unit='M'>{}</capacity>
				<target>
					<path>{}</path>
					<format type='qcow2'/>
					<permissions>
					</permissions>
					<timestamps>
					</timestamps>
					<compat>1.1</compat>
					<features>
						<lazy_refcounts/>
					</features>
				</target>
			</volume>
		)", disk_file_name(name), size, disk_path(name).generic_string()).c_str());

		auto pool = qemu_connect.storage_pool_lookup_by_name("testo-storage-pool");
		auto volume = pool.volume_create_xml(xml_config, {VIR_STORAGE_VOL_CREATE_PREALLOC_METADATA});
	}
}

void QemuVM::import_disk(const std::string& name, const fs::path& source) {
	if (use_external_snapshots()) {
		create_img_with_backing_file(disk_path(name), source);
	} else {
		fs_copy(source, disk_path(name));
	}

	auto pool = qemu_connect.storage_pool_lookup_by_name("testo-storage-pool");
	pool.refresh();
}

void QemuVM::create_disks() {
	try {
		if (use_external_snapshots()) {
			fs::path nvram_src = source_nvram_path();
			fs::path nvram_dst = current_nvram_path();
			fs_copy(nvram_src, nvram_dst);
		}
		if (config.count("disk")) {
			auto& disks = config.at("disk");
			for (size_t i = 0; i < disks.size(); ++i) {
				auto& disk = disks[i];
				std::string disk_name = disk.at("name");

				if (disk.count("source")) {
					fs::path source_disk = disk.at("source").get<std::string>();
					import_disk(disk_name, source_disk);
				} else {
					create_new_disk(disk_name, disk.at("size").get<uint32_t>());
				}
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Creating disks"));
	}
}

#ifdef __x86_64__
std::string QemuVM::preferable_video_model(const vir::Connect& qemu_connect) {
	auto dom_caps = qemu_connect.get_domain_capabilities();
	auto models_node = dom_caps.first_child().child("devices").child("video").child("enum");

	std::set<std::string> models;
	std::vector<std::string> preferable = {
		"vmvga",
		"qxl",
		"cirrus"
	};

	for (auto model = models_node.child("value"); model; model = model.next_sibling("value")) {
		models.insert(model.text().as_string());
	}

	for (auto& model: preferable) {
		if (models.find(model) != models.end()) {
			return model;
		}
	}

	throw std::runtime_error("Can't find any acceptable video model");
}
#endif

#ifdef __aarch64__
std::string QemuVM::preferable_video_model(const vir::Connect&) {
	return "virtio";
}
#endif

std::string QemuVM::mouse_button_to_str(MouseButton btn) {
	switch (btn) {
		case Left: return "left";
		case Right: return "right";
		case Middle: return "middle";
		case WheelUp: return "wheel-up";
		case WheelDown: return "wheel-down";
		default: throw std::runtime_error("Unknown button: " + btn);
	}
}

bool QemuVM::use_external_snapshots() const {
#ifdef __aarch64__
	return true;
#else
	return false;
#endif
}

fs::path QemuVM::disks_dir() const {
	auto pool = qemu_connect.storage_pool_lookup_by_name("testo-storage-pool");
	return pool.path();
}

fs::path QemuVM::memory_dir() const {
	return "/var/lib/libvirt/testo/ram/";
}

fs::path QemuVM::nvram_dir() const {
	return "/var/lib/libvirt/testo/nvram/";
}


fs::path QemuVM::current_nvram_path() const {
	return nvram_dir() / (id() + "@nvram");
}

fs::path QemuVM::nvram_snapshot_path(const std::string& snapshot_name) const {
	return nvram_dir() / (id() + "@nvram." + snapshot_name);
}

fs::path QemuVM::source_nvram_path() const {
	if (config.count("nvram")) {
		if (config.at("nvram").count("source")) {
			return config.at("nvram").at("source").get<std::string>();
		}
	}
	return "/usr/share/AAVMF/AAVMF_VARS.fd";
}

fs::path QemuVM::memory_snapshot_path(const std::string& snapshot_name) const {
	return memory_dir() / (id() + "@ram." + snapshot_name);
}

std::string QemuVM::disk_file_name(const std::string& name) const {
	return id() + "@" + name  + ".img";
}

fs::path QemuVM::disk_path(const std::string& name) const {
	return disks_dir() / disk_file_name(name);
}

std::string QemuVM::disk_snapshot_file_name(const std::string& disk_name, const std::string& snapshot_name) const {
	return id() + "@" + disk_name + ".img." + snapshot_name;
}

fs::path QemuVM::disk_snapshot_path(const std::string& disk_name, const std::string& snapshot_name) const {
	return disks_dir() / disk_snapshot_file_name(disk_name, snapshot_name);
}
