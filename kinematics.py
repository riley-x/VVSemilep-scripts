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
[utils.Sample.file_stubs]. For example,

    hists/{lep}lep/{sample}.root
    
See [utils.FileManager] for details.
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


@run
def plot_MC_backgrounds_per_region(file_manager : utils.FileManager, output_dir : str):
    '''
    Plots each sample in 'PE' mode, unstacked, to compare their relative distributions 
    and sizes. A ratio subplot shows their relative contribution.

    Repeats for each inclusive HP region and variable.
    '''
    vars : list[utils.Variable] = [
        utils.Variable.vvJ_m,
        utils.Variable.lvJ_m,
        utils.Variable.llJ_m,
    ]

    for lep in [0, 1, 2]:
        for region in [f'MergHP_Inclusive_{x}' for x in ['SR', 'TCR', 'MCR']]:
            for var in vars:
                ### Get hists ###
                hists = []
                legend = []
                for sample in file_manager.samples.values():
                    if 'data' in sample.name: continue
                    h = file_manager.get_hist(lep, sample.name, f'{{sample}}_VV{lep}Lep_{region}_{var}')
                    if h is None: continue
                    h = plot.rebin(h, var.rebin)
                    hists.append(h)
                    legend.append(sample.title)
                if not hists: continue

                ### Clean negative bins ###
                for h in hists:
                    for i in range(len(h)):
                        if h[i] < 0:
                            h[i] = 0
                            h.SetBinError(i, 0)

                ### Create ratio ###
                h_total = hists[0].Clone()
                for h in hists[1:]:
                    h_total.Add(h)
                
                ratios = []
                for h in hists:
                    r = h.Clone()
                    r.Divide(h_total)
                    r.Scale(100)
                    ratios.append(r)

                ### Plot ###
                plot.plot_ratio(
                    filename=f'{output_dir}/mc_bkg.{lep}lep.{region}.{var}',
                    **var.x_plot_opts(),

                    ### Titles ###
                    subtitle=[
                        '#sqrt{s}=13 TeV, 140 fb^{-1}',
                        f'{lep}Lep {region.replace("_", " ")}',
                    ],
                    legend=legend,

                    ### Main plot ###
                    ytitle='Events',
                    objs1=hists,
                    linecolor=plot.colors.tableu,
                    markersize=0,
                    opts='PE',

                    ### Ratio ###
                    height1=0.6,
                    ytitle2='% of total',
                    objs2=ratios,
                    opts2='HIST',
                    stack2=True,
                    fillcolor2=plot.colors.tableu_40,
                    y_range2=[0, 100],
                    outlier_arrows=False,
                )





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
    

def main():
    args = parse_args()
    file_manager = get_files(args.filepaths)

    plot.save_transparent_png = False
    plot.file_formats = ['png', 'pdf']

    for k,v in RUN_FUNCS.items():
        v(file_manager, args.output)


if __name__ == "__main__":
    main()