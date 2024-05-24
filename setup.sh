#!/bin/bash

if [ -f "ResonanceFinder/setup_RF_v21.sh" ]; then
    cd ResonanceFinder
    . setup_RF_v21.sh
    cd ..
else
    setupATLAS
    lsetup "root recommended"
    lsetup "python recommended"
fi