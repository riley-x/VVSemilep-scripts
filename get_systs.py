#!/usr/bin/env python3
'''
@file get_systs.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date March 19, 2024 
@brief Gets a list of the systematic variations

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------

    get_systs.py filepath/formatter_1.root [...] 

This will fetch histogram files using the naming convention supplied in the arguments.
These arguments can include python formatters (using curly braces) for 'lep' and 'sample'.
For example,

    hists/{lep}_{sample}.root
    
See [utils.FileManager] for details.
'''

import ROOT # type: ignore

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import utils


def get_systs(file_manager : utils.FileManager):
    f_template = file_manager.files[(1, utils.Sample.diboson.name)][0]
    start_key = 'SMVV_VV1Lep_MergHP_Inclusive_SR_lvJ_m_'
    end_key = '__1up'

    systs = []
    for key in f_template.GetListOfKeys():
        name = key.GetName()
        if name.startswith(start_key) and name.endswith(end_key):
            systs.append(name[len(start_key):-len(end_key)])

    systs.sort()
    return systs


##########################################################################################
###                                        MAIN                                        ###
##########################################################################################

def parse_args():
    parser = ArgumentParser(
        description="Sums MC to create an asimov data sample.", 
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('filepaths', nargs='+')
    return parser.parse_args()


def get_files(filepaths):
    file_manager = utils.FileManager(
        samples=[utils.Sample.diboson],
        file_path_formats=filepaths,
    )
    return file_manager


def main():
    args = parse_args()
    file_manager = get_files(args.filepaths)
    systs = get_systs(file_manager)

    for x in systs:
        print(f"'{x}',")
    

if __name__ == "__main__":
    main()