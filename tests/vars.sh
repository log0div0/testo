
set -euo pipefail

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
TESTO_SRC_DIR=$(readlink -f "$SCRIPT_DIR/..")
TESTO_BUILD_DIR=$(readlink -f "$SCRIPT_DIR/../build/out")
ISO_DIR=$(readlink -f "$TESTO_SRC_DIR/../iso")
LICENSE_PATH=$(readlink -f "$TESTO_SRC_DIR/../testo_license")
