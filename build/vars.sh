
set -euo pipefail

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
TESTO_SRC_DIR=$(readlink -f "$SCRIPT_DIR/..")
OUT_DIR=$(readlink -f "$SCRIPT_DIR/out")
TMP_DIR=$(readlink -f "$SCRIPT_DIR/tmp")
ISO_DIR=$(readlink -f "$TESTO_SRC_DIR/../iso")
TESTO_TESTS_DIR=$(readlink -f "$TESTO_SRC_DIR/tests/scripts/")
LICENSE_PATH=$(readlink -f "$TESTO_SRC_DIR/../testo_license")
