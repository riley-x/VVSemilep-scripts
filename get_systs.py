#!/usr/bin/env python3
'''
@file get_systs.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date March 19, 2024 
@brief Gets a list of the systematic variations

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------

    get_systs.py hist_file.root

'''

import ROOT # type: ignore

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import utils


def get_systs(filepath):
    f_template = ROOT.TFile(filepath)
    name_stub = '_MergHP_Inclusive_SR_'
    end_key = '__1up'

    systs = set()
    for key in f_template.GetListOfKeys():
        name = key.GetName()
        if name_stub in name and name.endswith(end_key):
            i = name.index('_Sys')
            systs.add(name[i+1:-len(end_key)])

    systs = list(systs)
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

def main():
    args = parse_args()
    systs = get_systs(args.filepaths[0])

    for x in systs:
        print(f"'{x}',")
    

if __name__ == "__main__":
    main()