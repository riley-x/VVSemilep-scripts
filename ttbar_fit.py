#!/usr/bin/env python3
'''
@file ttbar_fit.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date February 21, 2024 
@brief Script for fitting the ttbar/stop backgrounds using the 1lep channel.

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

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------

    ttbar_fit.py filepath/formatter_1.root [...]

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



##########################################################################################
###                                         RUN                                        ###
##########################################################################################


# def run_fit(
#         file_manager : utils.FileManager,
#     ):

#     x = ROOT.RooRealVar("x", "x", 0, 1)
    
#     n_stop_nom = 100
#     mu_stop_mean = ROOT.RooRealVar('mu_stop_mean', 'mu_stop_mean', 1 * n_stop_nom)
#     mu_stop_sigma = ROOT.RooRealVar('mu_stop_sigma', 'mu_stop_sigma', 0.2 * n_stop_nom)
#     mu_stop_alpha = ROOT.RooRealVar('mu_stop_alpha', 'mu_stop_alpha', n_stop_nom, 0, 2 * n_stop_nom)
#     mu_stop = ROOT.RooGaussian('mu_stop', 'mu_stop', mu_stop_alpha, mu_stop_mean, mu_stop_sigma)

#     n_ttbar_nom = 100
#     mu_ttbar = ROOT.RooRealVar('mu_ttbar', 'mu_ttbar', 100, 0, 200)
    
#     n_ttbar = ROOT.RooUniform('n_ttbar', 'n_ttbar', x)
#     n_stop = ROOT.RooUniform('n_stop', 'n_stop', x)
    
#     sig = ROOT.RooAddPdf("sig", "Signal", [n_ttbar, n_stop], [mu_stop, mu_ttbar])
    
    
#     data = ROOT.RooDataSet('data', 'data', [x])
#     data.add([x])
#     sig.createNLL(data)


def run_fit(
        file_manager : utils.FileManager,
    ):
    from scipy import optimize, stats


    y = 100
    n = 100

    def nll(params):
        mu = params[0]
        pred = mu * n
        return -stats.poisson.logpmf(y, pred) \
            - stats.norm.logpdf(mu, loc=1, scale=0.1) 
        
        # return -stats.norm.logpdf(y, loc=pred, scale=pred**0.5) \
        #     + pred
            # - stats.norm.logpdf(mu, loc=1, scale=0.1) \
    

    res = optimize.minimize(nll, [1.8], bounds=[(1e-2, 2)], method='L-BFGS-B')#, options={'ftol': 1e-15, 'gtol': 1e-15})
    print(res)




    


##########################################################################################
###                                        MAIN                                        ###
##########################################################################################

def parse_args():
    parser = ArgumentParser(
        description="Master run script for doing the unfolded analysis for VVsemilep.", 
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
    # args = parse_args()
    # file_manager = get_files(args.filepaths)

    # plot.save_transparent_png = False
    # plot.file_formats = ['png', 'pdf']

    run_fit(None)


if __name__ == "__main__":
    main()