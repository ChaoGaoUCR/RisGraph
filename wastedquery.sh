#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Run all commands
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/sx/stack_snap" "/home/cgao037/RisGraph/source/sx.txt" 1 0.01
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/sx/stack_snap" "/home/cgao037/RisGraph/source/sx.txt" 1 0.02
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/sx/stack_snap" "/home/cgao037/RisGraph/source/sx.txt" 1 0.04
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/sx/stack_snap" "/home/cgao037/RisGraph/source/sx.txt" 1 0.08
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/sx/stack_snap" "/home/cgao037/RisGraph/source/sx.txt" 1 0.16

"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/ttw/ttw_snap" "/home/cgao037/RisGraph/source/TW_queries.txt" 1 0.01
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/ttw/ttw_snap" "/home/cgao037/RisGraph/source/TW_queries.txt" 1 0.02
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/ttw/ttw_snap" "/home/cgao037/RisGraph/source/TW_queries.txt" 1 0.04
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/ttw/ttw_snap" "/home/cgao037/RisGraph/source/TW_queries.txt" 1 0.08
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/ttw/ttw_snap" "/home/cgao037/RisGraph/source/TW_queries.txt" 1 0.16

"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/or/snap_or" "/home/cgao037/RisGraph/source/or.txt" 1 0.01
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/or/snap_or" "/home/cgao037/RisGraph/source/or.txt" 1 0.02
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/or/snap_or" "/home/cgao037/RisGraph/source/or.txt" 1 0.04
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/or/snap_or" "/home/cgao037/RisGraph/source/or.txt" 1 0.08
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/or/snap_or" "/home/cgao037/RisGraph/source/or.txt" 1 0.16

"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/dl/snap_dl" "/home/cgao037/RisGraph/source/or.txt" 1 0.01
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/dl/snap_dl" "/home/cgao037/RisGraph/source/or.txt" 1 0.02
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/dl/snap_dl" "/home/cgao037/RisGraph/source/or.txt" 1 0.04
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/dl/snap_dl" "/home/cgao037/RisGraph/source/or.txt" 1 0.08
"$SCRIPT_DIR/build/wastedquery" "/home/cgao037/graph/dl/snap_dl" "/home/cgao037/RisGraph/source/or.txt" 1 0.16











