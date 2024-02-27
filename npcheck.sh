#!/bin/bash

set -e
set -u

mode="$1"
ws_path="$2"

if [ "$1" = "fits" ]; then
    cd ResonanceFinder/NPCheck
    ./runFitCrossCheck.py "$wsNPPath"
    cd ../..
elif [ "$1" = "pulls" ]; then
    # this requires the fits to be run first, uses default path fccs/FitCrossChecks.root
    cd ResonanceFinder/NPCheck
    root drawPullPlot.C
    cd ../..
else
    echo "$0 unknown mode: $1"
    exit 1
fi