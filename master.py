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

Check [unfolding.get_bins] and [gpr.FitConfig.get_bins_y] to ensure the desired binnings
are set.

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
import gpr
import ttbar_fit

##########################################################################################
###                                         RUN                                        ###
##########################################################################################


def run_gpr(
        file_manager : utils.FileManager, 
        lepton_channel : int,
        var: utils.Variable,
        output_dir: str,
        mu_stop : tuple[float, float],
        ttbar_fitter: ttbar_fit.TtbarSysFitter,
        from_csv_only : bool,
    ):
    ### Config ###
    config_base = {
        'lepton_channel': lepton_channel,
        'var': var,
        'output_dir': output_dir,
    }
    
    ### Nominal ###
    config = gpr.FitConfig(
        variation='nominal',
        mu_stop=mu_stop[0],
        mu_ttbar=ttbar_fitter.mu_ttbar_nom[0],
        **config_base,
    )
    gpr.run(file_manager, config, from_csv_only)

    ### Diboson signal strength variations ###
    for mu_diboson in [0.9, 0.95, 1.05, 1.1]:
        config = gpr.FitConfig(
            variation=f'mu-diboson{mu_diboson}',
            mu_stop=mu_stop[0],
            mu_ttbar=ttbar_fitter.mu_ttbar_nom[0],
            **config_base,
        )
        gpr.run(file_manager, config, from_csv_only)

    ### ttbar signal strength variations ###
    for variation in ['mu-ttbar_up', 'mu-ttbar_down']:
        config = gpr.FitConfig(
            variation=variation,
            mu_stop=mu_stop[0],
            mu_ttbar=ttbar_fitter.get_var(variation),
            **config_base,
        )
        gpr.run(file_manager, config, from_csv_only)

    ### stop signal strength variations ###
    config = gpr.FitConfig(
        variation='mu-stop_up',
        mu_stop=mu_stop[0] + mu_stop[1],
        mu_ttbar=ttbar_fitter.get_var('mu-stop_up'),
        **config_base,
    )
    gpr.run(file_manager, config, from_csv_only)
    config = gpr.FitConfig(
        variation=f'mu-stop_down',
        mu_stop=mu_stop[0] - mu_stop[1],
        mu_ttbar=ttbar_fitter.get_var('mu-stop_down'),
        **config_base,
    )
    gpr.run(file_manager, config, from_csv_only)

    # TODO syst variations
    

def run_channel(
        file_manager : utils.FileManager, 
        lepton_channel : int,
        mu_stop : tuple[float, float],
        ttbar_fitter : ttbar_fit.TtbarSysFitter,
        output_dir: str,
        from_csv_only : bool,
    ):
    log_base = f'master.py::run_channel({lepton_channel}lep)'
    vars = [utils.Variable.vv_m]

    ### Generate response matricies ###
    # plot.notice(f'{log_base} creating response matrix')
    # response_matrix_filepath = unfolding.main(
    #     file_manager=file_manager,
    #     sample=utils.Sample.diboson,
    #     lepton_channel=lepton_channel,
    #     output=f'{args.output}/response_matrix',
    #     vars=vars,
    # )

    ### Run GPR fit ###
    for var in vars:
        plot.notice(f'{log_base} running GPR fits for {var}')
        run_gpr(
            file_manager=file_manager,
            lepton_channel=lepton_channel,
            var=var,
            output_dir=f'{output_dir}/gpr',
            mu_stop=mu_stop,
            ttbar_fitter=ttbar_fitter,
            from_csv_only=from_csv_only,
        )


    ### Profile likelihood unfolding fit ###


    ### Likelihood fit (diboson signal strength) ###
        


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
    parser.add_argument('--from-csv-only', action='store_true', help="Don't do the GPR fit, just use the fit results in the CSV. This file should be placed at '{output}/gpr/gpr_fit_results.csv'.")
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

    ### ttbar fitter ###
    mu_stop = (1, 0.2)
    ttbar_fitter = ttbar_fit.TtbarSysFitter(file_manager, mu_stop_0=mu_stop)

    ### Loop over all channels ###
    for lepton_channel in [1]:
        run_channel(
            file_manager=file_manager,
            lepton_channel=lepton_channel,
            mu_stop=mu_stop, 
            ttbar_fitter=ttbar_fitter,
            output_dir=args.output,
            from_csv_only=args.from_csv_only,
        )
    


if __name__ == "__main__":
    main()