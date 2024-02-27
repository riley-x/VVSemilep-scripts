#!/bin/bash
# Run commands for NPCheck postscripts. 

set -e
set -u

ws_path="$1"
mode="$2"

if [ "$mode" = "fits" ]; then
    cd ResonanceFinder/NPCheck
    ./runFitCrossCheck.py "$ws_path"
    cd ../..
elif [ "$mode" = "drawFit" ]; then
    # this requires the fits to be run first, uses default path fccs/FitCrossChecks.root
    cd ResonanceFinder/NPCheck
    ./runDrawFit.py "$ws_path" --mu 1 --fccs 'fccs/FitCrossChecks.root'
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