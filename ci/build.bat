
call %~dp0\vars.bat

if "%~1"=="" (
	set TEST_SPEC=
) else (
	set TEST_SPEC=--test_spec %~1
)

testo run %SCRIPT_DIR%\build_scripts ^
	--stop_on_fail ^
	--prefix tb_ ^
	--param ISO_DIR %ISO_DIR% ^
	--param TESTO_SRC_DIR %TESTO_SRC_DIR% ^
	--param BUILD_ASSETS_DIR %SCRIPT_DIR%/build_assets ^
	--param TMP_DIR %TMP_DIR% ^
	--param OUT_DIR %OUT_DIR% ^
	--param ONNXRUNTIME_SRC_DIR %ONNXRUNTIME_SRC_DIR% ^
	--param WIN10_TEMPLATE_PATH "D:\HyperV Disks\testo-builder-win10-template.vhdx" ^
	--license %LICENSE_PATH% ^
	%TEST_SPEC% ^
	--assume_yes