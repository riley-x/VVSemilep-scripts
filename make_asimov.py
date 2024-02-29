#!/usr/bin/env python3
'''
@file make_asimov.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date February 29, 2024 
@brief Sums MC to create an asimov data sample

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------

    make_asimov.py filepath/formatter_1.root [...] \
        --output {lep}lep_data-asimov.hists.root

This will fetch histogram files using the naming convention supplied in the arguments.
These arguments can include python formatters (using curly braces) for 'lep', which will
be replaced with the lepton channel number, and 'sample', which uses
[utils.Sample.file_stubs]. For example,

    hists/{lep}lep/{sample}.root
    
See [utils.FileManager] for details.
'''

import ROOT

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import utils


##########################################################################################
###                                        MAIN                                        ###
##########################################################################################

def parse_args():
    parser = ArgumentParser(
        description="Sums MC to create an asimov data sample.", 
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('filepaths', nargs='+')
    parser.add_argument('-o', '--output', default='{lep}lep_data-asimov.hists.root')
    return parser.parse_args()


def get_files(filepaths):
    file_manager = utils.FileManager(
        samples=[
            utils.Sample.wjets,
            utils.Sample.zjets,
            utils.Sample.ttbar,
            utils.Sample.stop,
            utils.Sample.diboson,
        ],
        file_path_formats=filepaths,
    )
    return file_manager


def main():
    args = parse_args()
    file_manager = get_files(args.filepaths)

    for lepton_channel in [0, 1, 2]:
        f_template = file_manager.files[(lepton_channel, utils.Sample.diboson.name)][0]
        f_out = ROOT.TFile(args.output.format(lep=lepton_channel), 'RECREATE')
        for key in f_template.GetListOfKeys():
            name = key.GetName()
            if 'MergHP_Inclusive' not in name and \
                '__v__fatjet_m' not in name and \
                'unfoldingMtx' not in name:
                continue
            name = name.replace(utils.Sample.diboson.hist_keys[0], '{sample}')
            print(name)

            hists = file_manager.get_hist_all_samples(lepton_channel, name)
            h_data = None
            for _,h in hists.items():
                if h_data is None:
                    h_data = h.Clone()
                else:
                    h_data.Add(h)
            h_data.SetName(name.format(sample='data'))
            h_data.Write()
        f_out.Close()
    

if __name__ == "__main__":
    main()