#!/bin/bash

set -e
set -u

ws_path="$1"
mode="$2"

if [ "$mode" = "fits" ]; then
    cd ResonanceFinder/NPCheck
    ./runFitCrossCheck.py "$wsNPPath"
    cd ../..
elif [ "$mode" = "pulls" ]; then
    # this requires the fits to be run first, uses default path fccs/FitCrossChecks.root
    cd ResonanceFinder/NPCheck
    root drawPullPlot.C
    cd ../..
else
    echo "$0 unknown mode: $mode"
    exit 1
fi