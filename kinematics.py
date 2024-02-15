#!/usr/bin/env python3
'''
@file kinematics.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date February 14, 2024 
@brief Simple kinematic plots from reader histograms

------------------------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------------------------

    setupATLAS 
    lsetup "root recommended" 
    lsetup "python centos7-3.9"

Note that this can't be setup at the same time with AnalysisBase or else you get a lot of
conflicts :(

------------------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------------------
This script will generate a lot of plots, using each of the individual functions in the
PLOT block below. If you want to remove specific plots, simply delete the `@run` decorator
from the function.

Check [utils.Sample] to make sure the hardcoded naming stuctures are correct.

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------

    kinematics.py filepath/formatter_1.root [...]

This will fetch files using the naming convention supplied in the arguments. These
arguments can include python formatters (using curly braces) for 'lep', which will be
replaced with the lepton channel number, and 'sample', which uses
[utils.Sample.file_stubs]. For example, the argument could be
`hists/{lep}lep/{sample}.root`.
'''

from plotting import plot
import ROOT
import numpy as np
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import utils

RUN_FUNCS = {}
def run(f):
    '''
    Decorator that registers the input function to be executed
    '''
    RUN_FUNCS[f.__name__] = f
    return f


##########################################################################################
###                                        PLOT                                        ###
##########################################################################################





##########################################################################################
###                                        MAIN                                        ###
##########################################################################################

def parse_args():
    parser = ArgumentParser(
        description="Plots various kinematic comparisons between the MC backgrounds and data for the VVSemileptonic analysis.", 
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('filepaths', nargs='+')
    return parser.parse_args()


def get_files(filepaths):
    file_manager = utils.FileManager(
        samples=[
            utils.Sample.wjets,
            utils.Sample.zjets,
            utils.Sample.ttbar,
            utils.Sample.stop,
            utils.Sample.diboson,
            utils.Sample.data,
        ],
        file_path_formats=filepaths,
    )
    print(file_manager.files)
    

def main():
    args = parse_args()
    file_manager = get_files(args.filepaths)

    for k,v in RUN_FUNCS.items():
        v()


if __name__ == "__main__":
    main()