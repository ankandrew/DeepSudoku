#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# Apply formatters
"${SCRIPT_DIR}/apply_formatters.sh"
# Run linters
"${SCRIPT_DIR}/run_linters.sh"
