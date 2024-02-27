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
conflicts :(. Alternatively, you can just setup ResonanceFinder.

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
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from plotting import plot
import utils
import unfolding
import gpr
import ttbar_fit
import diboson_fit

##########################################################################################
###                                        PLOTS                                       ###
##########################################################################################

def plot_gpr_mu_diboson_correlations(
        config : gpr.FitConfig,
        yields : list[float],
        filename : str,
    ):
    '''
    This plots the change in yields as the diboson signal strength guess is varied. This
    function also returns the average delta_N/delta_mu for each bin in [config.bins_y].
    '''
    ### Get graphs ###
    csv_base_spec = {
        'lep': config.lepton_channel, 
        'fitter': f'{config.gpr_version}_marg_post',
        'vary': config.var.name, 
        'bins': config.bins_y,
    }
    i_nom = len(yields) // 2
    graphs = [config.fit_results.get_graph(**csv_base_spec, variation=f'mu-diboson{x}', unscale_width=True) for x in yields]
    graphs.insert(i_nom, config.fit_results.get_graph(**csv_base_spec, variation='nominal', unscale_width=True))
    legend = [str(x) for x in yields]
    legend.insert(i_nom, '1.0')

    ### Get ratios ###
    average_deltas = np.zeros(len(config.bins_y) - 1)
    h_nom = graphs[i_nom]
    def subplot(hists, **kwargs):
        # Use the callback here to copy the formatting from [hists]
        ratios = []
        for h,mu in zip(hists, legend):
            mu = float(mu)
            if mu == 1: continue

            h = h.Clone()
            for i in range(h.GetN()):
                delta_delta = (h.GetPointY(i) - h_nom.GetPointY(i)) / h_nom.GetPointY(i) / (1 - mu)
                h.SetPointY(i, delta_delta)
                h.SetPointEYhigh(i, 0)
                h.SetPointEYlow(i, 0)
                average_deltas[i] += delta_delta
            ratios.append(h)
        return ratios

    ### Plot ###
    gpr.plot_summary_distribution(
        hists=graphs,
        filename=filename,
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel',
            'Scanning #mu^{diboson}_{guess}',
        ],
        legend=legend,
        ytitle='Events',
        ytitle2='#frac{#DeltaFit / Fit_{nom}}{#Delta#mu}',
        xtitle=f'{config.var:title}',
        edge_labels=[str(x) for x in config.bins_y],
        subplot2=subplot,
        subplot3=None,
        opts2='P',
        y_range2=[0, 0.18],
        ydivs2=503,
    )

    return average_deltas / len(yields)


def plot_gpr_ttbar_and_stop_correlations(config : gpr.FitConfig, filename : str):
    '''
    This plots the change in yields as the ttbar and stop signal strengths are varied. 
    '''
    ### Get graphs ###
    csv_base_spec = {
        'lep': config.lepton_channel, 
        'fitter': f'{config.gpr_version}_marg_post',
        'vary': config.var.name, 
        'bins': config.bins_y,
    }
    variations = ['nominal', 'mu-ttbar_up', 'mu-ttbar_down', 'mu-stop_up', 'mu-stop_down']
    graphs = [config.fit_results.get_graph(**csv_base_spec, variation=x, unscale_width=True) for x in variations]

    gpr.plot_summary_distribution(
        hists=graphs,
        filename=filename,
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel',
        ],
        legend=variations,
        ytitle='Events',
        ytitle2='Var / Nom',
        xtitle=f'{config.var:title}',
        edge_labels=[str(x) for x in config.bins_y],
        subplot2='ratios',
        subplot3=None,
        y_range2=[0.9, 1.1],
        ydivs2=503,
    )


def plot_diboson_yield(h_fit, h_mc, **plot_opts):
    h_mc.SetMarkerSize(0)
    h_mc.SetLineColor(plot.colors.blue)

    h_mc_err = h_mc.Clone()
    h_mc_err.SetFillColorAlpha(plot.colors.blue, 0.3)

    ratio = h_fit.Clone()
    ratio.Divide(h_mc)

    plot.plot_ratio(
        objs1=[h_mc, h_mc_err, h_fit],
        objs2=[ratio],
        legend=['', 'MC (stat only)', 'Fit'],
        opts=['HIST', 'E2', 'PE'],
        legend_opts=['', 'FL', 'PEL'],
        opts2='PE',
        ytitle='Events',
        ytitle2='Fit / MC',
        hline=1,
        **plot_opts,
    )

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
    ) -> tuple[gpr.FitResults, list[float]]:
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
    # raise RuntimeError('asdf')

    ### Diboson signal strength variations ###
    mu_diboson_points = [0.9, 0.95, 1.05, 1.1]
    for mu_diboson in mu_diboson_points:
        config = gpr.FitConfig(
            variation=f'mu-diboson{mu_diboson}',
            mu_stop=mu_stop[0],
            mu_ttbar=ttbar_fitter.mu_ttbar_nom[0],
            **config_base,
        )
        gpr.run(file_manager, config, from_csv_only)
    mu_diboson_corrs = plot_gpr_mu_diboson_correlations(
        config=config, 
        yields=mu_diboson_points,
        filename=f'{output_dir}/{config.lepton_channel}lep/{config.var}/gpr_diboson_mu_scan',
    )

    ### ttbar signal strength variations ###
    for variation in ['mu-ttbar_up', 'mu-ttbar_down']:
        config = gpr.FitConfig(
            variation=variation,
            mu_stop=mu_stop[0],
            mu_ttbar=ttbar_fitter.get_var(variation),
            **config_base,
        )
        gpr.run(file_manager, config, from_csv_only)

    ### Single top signal strength variations ###
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
    plot_gpr_ttbar_and_stop_correlations(config, f'{output_dir}/{config.lepton_channel}lep/{config.var}/gpr_ttbar_stop_mu_scan')

    # TODO syst variations


    ### Create histograms for ResonanceFinder ###
    

    return config.fit_results, mu_diboson_corrs
    

def save_rebinned_histograms(
        file_manager : utils.FileManager, 
        lepton_channel : int,
        output_dir : str,
        variables : list[utils.Variable],
    ):

    output_files = {}
    def file(sample):
        if sample not in output_files:
            output_files[sample] = ROOT.TFile(f'{output_dir}/{lepton_channel}lep_{sample}_rebin.root', 'RECREATE')
        return output_files[sample]
    
    variations = [utils.variation_nom] + utils.variations_hist
    for variable in variables:
        bins = utils.get_bins(lepton_channel, variable)
        bins = np.array(bins, dtype=float)
        for variation in variations:
            hist_name = '{sample}_VV1Lep_MergHP_Inclusive_SR_' + utils.generic_var_to_lep(variable, lepton_channel).name
            hist_name = utils.hist_name_variation(hist_name, variation)

            hists = file_manager.get_hist_all_samples(lepton_channel, hist_name)
            for sample,hist in hists.items():
                hist = plot.rebin(hist, bins)
                hist.SetName(hist_name.format(sample=sample))

                f = file(sample)
                f.cd()
                hist.Write()
    
    file_format = f'{output_dir}/{lepton_channel}lep_{{sample}}_rebin.root'
    plot.success(f'Saved rebinned histograms to {file_format}')
    return file_format


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
    plot.notice(f'{log_base} creating response matrix')
    response_matrix_filepath = unfolding.main(
        file_manager=file_manager,
        sample=utils.Sample.diboson,
        lepton_channel=lepton_channel,
        output=f'{output_dir}/response_matrix',
        vars=vars,
    )

    ### Rebin reco histograms ###
    plot.notice(f'{log_base} saving rebinned histograms')
    rebinned_histogram_path = save_rebinned_histograms(
        file_manager=file_manager,
        lepton_channel=lepton_channel,
        output_dir=output_dir,
        variables=vars,
    )

    ### Iterate per variable ###
    for var in vars:
        ### Bins ###
        bins = utils.get_bins(lepton_channel, var)

        ### GPR fit ###
        plot.notice(f'{log_base} running GPR fits for {var}')
        fit_results, gpr_mu_corr = run_gpr(
            file_manager=file_manager,
            lepton_channel=lepton_channel,
            var=var,
            output_dir=f'{output_dir}/gpr',
            mu_stop=mu_stop,
            ttbar_fitter=ttbar_fitter,
            from_csv_only=from_csv_only,
        )

        ### Diboson yield ###
        # So this will be superceded by the profile likelihood fit, which will use the
        # GPR fit results directly, but this is a nice check.
        h_diboson_fit = ROOT.TH1F('h_diboson', 'Diboson', len(bins) - 1, np.array(bins, dtype=float))
        h_diboson_mc = ROOT.TH1F('h_diboson', 'Diboson', len(bins) - 1, np.array(bins, dtype=float))
        for i in range(len(bins) - 1):
            res = diboson_fit.run_fit(
                file_manager=file_manager,
                gpr_results=fit_results,
                ttbar_fitter=ttbar_fitter,
                lepton_channel=lepton_channel,
                variable=var,
                bin=(bins[i], bins[i+1]),
                mu_stop=mu_stop,
                gpr_mu_corr=gpr_mu_corr[i],
            )
            diboson_yield = res['diboson-yield']
            h_diboson_fit.SetBinContent(i+1, diboson_yield[0])
            h_diboson_fit.SetBinError(i+1, diboson_yield[1])

            diboson_yield = res['diboson-yield-mc-statonly']
            h_diboson_mc.SetBinContent(i+1, diboson_yield[0])
            h_diboson_mc.SetBinError(i+1, diboson_yield[1])
        plot_diboson_yield(
            h_fit=h_diboson_fit, 
            h_mc=h_diboson_mc,
            filename=f'{output_dir}/{lepton_channel}lep_{var}_yields',
            subtitle=[
                '#sqrt{s}=13 TeV, 140 fb^{-1}',
                f'{lepton_channel}-lepton channel',
            ],
            xtitle=f'{var:title}',
        )

        ### Profile likelihood unfolding fit ###
        os.makedirs(f'{output_dir}/rf', exist_ok=True)
        try:
            ### Create RF workspace ###
            # Really important this import is here otherwise segfaults occur, due to the
            # `from ROOT import RF` line I think. But somehow hiding it here is fine.
            import rf_plu 
            import subprocess
            import shutil

            plot.notice(f'{log_base} creating ResonanceFinder workspace')
            ws_path = rf_plu.run(
                lepton_channel=lepton_channel,
                variable=var,
                nbins=len(bins) - 1,
                response_matrix_path=response_matrix_filepath,
                output_dir=output_dir,
                hist_file_format=rebinned_histogram_path,
                mu_stop=mu_stop,
                mu_ttbar=ttbar_fitter.mu_ttbar_nom,
            )
            ws_path = os.path.abspath(ws_path)

            ### Run fits ###
            plot.notice(f'{log_base} running PLU fits')
            npcheck_dir = 'ResonanceFinder/NPCheck'
            res = subprocess.run(['./runFitCrossCheck.py', ws_path], cwd=npcheck_dir) # the './' is necessary!
            res.check_returncode()

            ### Draw fit ###
            plot.notice(f'{log_base} drawing PLU fits')
            res = subprocess.run(
                ['./runDrawFit.py', ws_path, 
                    '--mu', '1', 
                    '--fccs', 'fccs/FitCrossChecks.root'
                ], 
                cwd=npcheck_dir
            )
            res.check_returncode()
            npcheck_output_path = f'{npcheck_dir}/Plots/PostFit/summary_postfit_doAsimov0_doCondtional0_mu1.pdf'
            target_path = f'{output_dir}/rf/postfit.pdf'
            shutil.copyfile(npcheck_output_path, target_path)

            print(res.stdout)

        except Exception as e:
            plot.warning(str(e))
        



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