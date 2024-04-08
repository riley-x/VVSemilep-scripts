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
        [--output {lep}lep_data-asimov.hists.root]

This will fetch histogram files using the naming convention supplied in the arguments.
These arguments can include python formatters (using curly braces) for 'lep', which will
be replaced with the lepton channel number, and 'sample', which uses
[utils.Sample.file_stubs]. For example,

    hists/{lep}lep/{sample}.root
    
See [utils.FileManager] for details.
'''

import ROOT # type: ignore

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
    parser.add_argument('-o', '--output', default='hist_data-asimov_{lep}lep-0.root')
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
            if '__1up' in name or '__1down' in name or '_Sys' in name:
                continue
            name = name.replace(utils.Sample.diboson.hist_keys[0], '{sample}')
            print(name)

            ### Sum hists ###
            hists = file_manager.get_hist_all_samples(lepton_channel, name)
            h_data = None
            for sample,h in hists.items():
                # print(sample.rjust(15), h.Integral())
                if h is None:
                    print(f'Warning! Missing histogram for {name} for {sample}')
                    continue
                elif h_data is None:
                    h_data = h.Clone()
                else:
                    h_data.Add(h)
            h_data.SetName(name.format(sample='data'))
            # print('Data'.rjust(15), h_data.Integral())

            ### Set sqrt(N) errors ###
            for i in range(len(h_data)):
                h_data.SetBinError(i, abs(h_data[i]) ** 0.5)
            h_data.Write()

        f_out.Close()
    

if __name__ == "__main__":
    main()