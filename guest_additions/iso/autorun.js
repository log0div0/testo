
var shell = WScript.CreateObject("WScript.Shell")
var arch = shell.ExpandEnvironmentStrings("%PROCESSOR_ARCHITECTURE%")

if (arch == "AMD64") {
	shell.Run("testo-guest-additions-x64.msi")
}

if (arch == "x86") {
	shell.Run("testo-guest-additions-x86.msi")
}
