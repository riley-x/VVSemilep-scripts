#!/usr/bin/env python3
'''
@file master.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date February 20, 2024 
@brief Master run script for doing the unfolded analysis for VVsemilep

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

Check [utils.Sample] and [utils.Variable] to make sure the hardcoded naming stuctures are
correct.

Check [unfolding.get_bins] to ensure the desired binnings are set.

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------

    main.py filepath/formatter_1.root [...]

This will fetch histogram files using the naming convention supplied in the arguments.
These arguments can include python formatters (using curly braces) for 'lep', which will
be replaced with the lepton channel number, and 'sample', which uses
[utils.Sample.file_stubs]. For example,

    hists/{lep}lep/{sample}.root
    
See [utils.FileManager] for details.
'''

import ROOT
import numpy as np
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from plotting import plot
import utils
import unfolding

##########################################################################################
###                                        MAIN                                        ###
##########################################################################################

def parse_args():
    parser = ArgumentParser(
        description="Plots various kinematic comparisons between the MC backgrounds and data for the VVSemileptonic analysis.", 
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('filepaths', nargs='+')
    parser.add_argument('-o', '--output', default='./output')
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
    return file_manager
    

def run_channel(args, file_manager : utils.FileManager, lepton_channel : int):
    log_base = f'master.py::run_channel({lepton_channel}lep)'

    ### Generate response matricies ###
    plot.notice(f'{log_base} creating response matrix')
    response_matrix_dir = f'{args.output}/response_matrix'
    response_matrix_filepath = unfolding.main(
        file_manager=file_manager,
        sample=utils.Sample.diboson,
        lepton_channel=lepton_channel,
        output=response_matrix_dir,
        vars=[utils.Variable.vv_m],
    )
    
    ### Run GPR fit ###


    ### Likelihood fit ###


def main():
    args = parse_args()
    file_manager = get_files(args.filepaths)

    plot.save_transparent_png = False
    plot.file_formats = ['png', 'pdf']

    for lepton_channel in [0, 1, 2]:
        run_channel(
            args=args,
            file_manager=file_manager,
            lepton_channel=lepton_channel,
        )
    


if __name__ == "__main__":
    main()