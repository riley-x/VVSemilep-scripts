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

from __future__ import annotations

import ROOT

import numpy as np
import os
import sys
import subprocess
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from plotting import plot
import utils
import unfolding
import gpr
import ttbar_fit
import diboson_fit


##########################################################################################
###                                        UTILS                                       ###
##########################################################################################

def convert_alpha(alpha : tuple[float, float], orig : tuple[float, float]) -> tuple[float, float]:
    '''
    For a Gaussian constrained floating parameter in the PLU fit, given a result w/ error 
    in terms of [alpha], returns the actual values.
    '''
    val = orig[0] + alpha[0] * orig[1]
    err = alpha[1] * orig[1]
    return (val, err)

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


def plot_yield_comparison(h_fit, h_mc, **plot_opts):
    '''
    Plots a 1D comparison between a fit and MC. The fit is shown as data points while the
    MC is shown as a filled blue band. Includes a fit/MC ratio.
    '''
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
        y_range2=(0.5, 1.5),
        **plot_opts,
    )


def plot_pre_plu_fit(config : ChannelConfig, variable : utils.Variable):
    '''
    Plots a stack plot comparing data to backgrounds prior to the PLU fit but using the
    GPR background estimate.
    '''
    ### Get hists ###
    f_gpr = ROOT.TFile(f'{config.output_dir}/gpr/gpr_{config.lepton_channel}lep_vjets_yield.root')
    h_gpr = f_gpr.Get('Vjets_SR_' + variable.name)
        
    hist_name = '{sample}_VV{lep}Lep_MergHP_Inclusive_SR_' + utils.generic_var_to_lep(variable, config.lepton_channel).name
    h_diboson = config.file_manager.get_hist(config.lepton_channel, utils.Sample.diboson, hist_name)
    h_ttbar = config.file_manager.get_hist(config.lepton_channel, utils.Sample.ttbar, hist_name)
    h_stop = config.file_manager.get_hist(config.lepton_channel, utils.Sample.stop, hist_name)
    h_data = config.file_manager.get_hist(config.lepton_channel, utils.Sample.data, hist_name)

    bins = utils.get_bins(config.lepton_channel, variable)
    h_diboson = plot.rebin(h_diboson, bins)
    h_ttbar = plot.rebin(h_ttbar, bins)
    h_stop = plot.rebin(h_stop, bins)
    h_data = plot.rebin(h_data, bins)

    h_ttbar.Scale(config.ttbar_fitter.mu_ttbar_nom[0])
    h_stop.Scale(config.mu_stop[0])

    h_sum = h_gpr.Clone()
    h_sum.Add(h_diboson)
    h_sum.Add(h_ttbar)
    h_sum.Add(h_stop)

    h_ratio = h_data.Clone()
    h_errs = h_sum.Clone()
    h_ratio.Divide(h_sum)
    h_errs.Divide(h_sum)

    ### Plot ###
    pads = plot.RatioPads()
    plotter1 = pads.make_plotter1(
        ytitle='Events',
        logy=True,
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel pre-PLU',
        ],
    )
    plotter1.add(
        objs=[h_diboson, h_stop, h_ttbar, h_gpr], 
        legend=['Diboson (#mu=1)', 'Single top', 't#bar{t}', 'GPR (V+jets)'],
        stack=True,
        opts='HIST',
        fillcolor=[plot.colors.pastel_blue, plot.colors.pastel_yellow, plot.colors.pastel_orange, plot.colors.pastel_red],
        linewidth=1,
    )
    plotter1.add(
        objs=[h_sum],
        fillcolor=plot.colors.gray,
        fillstyle=3145,
        linewidth=0,
        markerstyle=0,
        opts='E2',
        legend_opts='F',
        legend=['GPR err + MC stat'],
    )
    plotter1.add(
        objs=[h_data],
        legend=['Data'],
        opts='PE',
        legend_opts='PEL',
    )
    plotter1.draw()

    ### Subplot ###
    plotter2 = pads.make_plotter2(
        ytitle='Data / Bkgs',
        xtitle='m(J) [GeV]',
        ignore_outliers_y=False,
        y_range=(0.5, 1.5),
    )
    plotter2.add(
        objs=[h_errs],
        fillcolor=plot.colors.gray,
        fillstyle=3145,
        linewidth=0,
        markerstyle=0,
        opts='E2',
        legend=None,
    )
    plotter2.add(
        objs=[h_ratio],
        opts='PE',
        legend=None,
    )
    plotter2.draw()
    plotter2.draw_hline(1, ROOT.kDashed)

    plot.save_canvas(pads.c, f'{config.output_dir}/plots/{config.lepton_channel}lep_{variable}.plu_prefit')


def plot_plu_fit(config : ChannelConfig, variable : utils.Variable, fit_results : dict[str, tuple[float, float]]):
    '''
    Plots a stack plot comparing data to backgrounds after the PLU fit.
    '''
    bins = utils.get_bins(config.lepton_channel, variable)

    ######################################################################################
    ### HISTS
    ######################################################################################
    
    ### GPR ###
    f_gpr = ROOT.TFile(f'{config.output_dir}/gpr/gpr_{config.lepton_channel}lep_vjets_yield.root')
    h_gpr = f_gpr.Get('Vjets_SR_' + variable.name)

    ### Response matrix ###
    f_response_mtx = ROOT.TFile(config.response_matrix_filepath)
    h_signals = []
    for i in range(1, len(bins)):
        h = f_response_mtx.Get(f'ResponseMatrix_{variable}_fid{i:02}')
        h.Scale(fit_results[f'mu_{i:02}'][0])
        h_signals.append(h)
        
    ### MC + data ###
    hist_name = '{sample}_VV{lep}Lep_MergHP_Inclusive_SR_' + utils.generic_var_to_lep(variable, config.lepton_channel).name
    h_ttbar = config.file_manager.get_hist(config.lepton_channel, utils.Sample.ttbar, hist_name)
    h_stop = config.file_manager.get_hist(config.lepton_channel, utils.Sample.stop, hist_name)
    h_data = config.file_manager.get_hist(config.lepton_channel, utils.Sample.data, hist_name)

    h_ttbar = plot.rebin(h_ttbar, bins)
    h_stop = plot.rebin(h_stop, bins)
    h_data = plot.rebin(h_data, bins)

    h_ttbar.Scale(fit_results['mu-ttbar'][0])
    h_stop.Scale(fit_results['mu-stop'][0])

    ### Sum and ratio ###
    h_sum = h_gpr.Clone()
    h_sum.Add(h_ttbar)
    h_sum.Add(h_stop)
    for h in h_signals:
        h_sum.Add(h)

    h_ratio = h_data.Clone()
    h_errs = h_sum.Clone()
    h_ratio.Divide(h_sum)
    h_errs.Divide(h_sum)

    ######################################################################################
    ### PLOT
    ######################################################################################
    
    pads = plot.RatioPads()

    ### Main pad options ###
    plotter1 = pads.make_plotter1(
        text_pos='top',
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel post-PLU',
        ],
        legend_columns=2,
        legend_vertical_order=True,
        ytitle='Events',
        logy=True,
        y_min=0.2,
    )

    ### Colors ###
    palette_colors = plot.colors_from_palette(ROOT.kViridis, len(h_signals))
    palette_colors = [plot.whiteblend(ROOT.gROOT.GetColor(x), 0.4) for x in palette_colors]
    def stack_color(i):
        if i < len(h_signals):
            return palette_colors[i].GetNumber()
        else:
            mc_colors = [plot.colors.pastel_yellow, plot.colors.pastel_orange, plot.colors.pastel_red]
            return mc_colors[i - len(h_signals)]
        
    ### Legend ###
    legend = [f'Bin {i}' for i in range(1, len(bins))][::-1]
    legend += ['top', 't#bar{t}', 'V+jets']

    ### Plot stack ###
    plotter1.add(
        objs=h_signals[::-1] + [h_stop, h_ttbar, h_gpr], 
        legend=legend,
        stack=True,
        opts='HIST',
        fillcolor=stack_color,
        linewidth=1,
    )

    ### Plot error hash ###
    plotter1.add(
        objs=[h_sum],
        fillcolor=plot.colors.gray,
        fillstyle=3145,
        linewidth=0,
        markerstyle=0,
        opts='E2',
        legend_opts='F',
        legend=['Errs - syst'],
    )

    ### Plot data ###
    plotter1.add(
        objs=[h_data],
        legend=['Data'],
        opts='PE',
        legend_opts='PEL',
    )
    plotter1.draw()

    ### Subplot ###
    plotter2 = pads.make_plotter2(
        ytitle='Data / Fit',
        xtitle='m(J) [GeV]',
        ignore_outliers_y=False,
        y_range=[0.5, 1.5],
    )
    plotter2.add(
        objs=[h_errs],
        fillcolor=plot.colors.gray,
        fillstyle=3145,
        linewidth=0,
        markerstyle=0,
        opts='E2',
        legend=None,
    )
    plotter2.add(
        objs=[h_ratio],
        opts='PE',
        legend=None,
    )
    plotter2.draw()
    plotter2.draw_hline(1, ROOT.kDashed)

    plot.save_canvas(pads.c, f'{config.output_dir}/plots/{config.lepton_channel}lep_{variable}.plu_postfit')


def plot_pulls(config : ChannelConfig, variable : utils.Variable, fit_results : dict[str, tuple[float, float]], filename : str):
    all_vars = [utils.variation_lumi] + utils.variations_custom + utils.variations_hist
    nvars = len(all_vars)

    ### Create hist ###
    h = ROOT.TH1F('h_pulls', '', nvars, 0, nvars)
    for i,var in enumerate(all_vars):
        alpha = fit_results[f'alpha_{var}']
        h.SetBinContent(i + 1, alpha[0])
        h.SetBinError(i + 1, alpha[1])
        h.GetXaxis().SetBinLabel(i + 1, var)
    h.GetXaxis().LabelsOption('u')

    ### Plotter ###
    ROOT.gStyle.SetErrorX(0)
    ROOT.gStyle.SetEndErrorSize(5)
    c = ROOT.TCanvas('c1', 'c1', 1000, 800)
    
    plotter = plot.Plotter(
        pad=c,
        _frame=h,
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel {variable} pulls',
        ],
        ytitle='(#theta_{fit} - #theta_{0}) / #sigma_{#theta}',
        y_range=(-5, 5),
        bottom_margin=0.3,
    )

    ### Plot objects ###
    b1 = ROOT.TBox(0, -1, nvars, 1)
    b2 = ROOT.TBox(0, -2, nvars, 2)
    b1.SetFillColor(plot.colors.pastel_green)
    b2.SetFillColor(plot.colors.pastel_yellow)
    plotter.add_primitives([b2, b1])
    plotter.add([h], opts='E1', markersize=2, markerstyle=ROOT.kFullSquare)
    plotter.draw()

    plot.save_canvas(c, filename)
    ROOT.gStyle.SetErrorX()
    ROOT.gStyle.SetEndErrorSize(0)


def plot_correlations(config : ChannelConfig, variable : utils.Variable, roofit_results, filename : str):
    ### Create hist ###
    bins = utils.get_bins(config.lepton_channel, variable)
    alphas = [utils.variation_lumi] + utils.variations_custom + utils.variations_hist
    mus = [f'mu_{x:02}' for x in range(1, len(bins))]
    pars = [f'alpha_{x}' for x in alphas] + mus

    n = len(pars)
    h = ROOT.TH2F('h_cov', '', n, 0, n, n, 0, n)
    for i in range(n):
        h.GetXaxis().SetBinLabel(i + 1, pars[i].replace('alpha_', ''))
        h.GetYaxis().SetBinLabel(n - i, pars[i].replace('alpha_', ''))
        for j in range(n):
            h.SetBinContent(n - i, j + 1, roofit_results.correlation(pars[i], pars[j]))
    h.GetXaxis().LabelsOption('v')

    ### Plot ###
    plot.create_gradient(np.array([
        [0.0,  0.0, 0.2, 1.0],
        [0.1,  0.0, 0.5, 1.0],
        [0.45, 1.0, 1.0, 1.0],
        [0.55, 1.0, 1.0, 1.0],
        [0.9,  1.0, 0.5, 0.0],
        [1.0,  1.0, 0.2, 0.0],
    ]))
    ROOT.gStyle.SetNumberContours(101)
    plot.plot(
        filename=filename,
        text_pos='topright',
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel {variable} fit',
        ],
        objs=[h],
        opts='COLZ',
        z_range=[-1, 1],
        left_margin=0.3,
        bottom_margin=0.3,
    )



##########################################################################################
###                                       CONFIG                                       ###
##########################################################################################

class ChannelConfig:

    def __init__(
            self, 
            lepton_channel : int, 
            file_manager : utils.FileManager, 
            ttbar_fitter: ttbar_fit.TtbarSysFitter,
            mu_stop : tuple[float, float],
            output_dir : str,
            skip_fits : bool,
            skip_gpr : bool,
            is_asimov : bool,
        ):
        self.lepton_channel = lepton_channel
        self.file_manager = file_manager
        self.ttbar_fitter = ttbar_fitter
        self.mu_stop = mu_stop
        self.output_dir = output_dir
        self.skip_fits = skip_fits
        self.skip_gpr = skip_gpr
        self.is_asimov = is_asimov

        self.log_base = f'master.py::run_channel({lepton_channel}lep)'
        self.variables = [utils.Variable.vv_m]
        self.npcheck_dir = 'ResonanceFinder/NPCheck'

        ### Set in run_channel ###
        self.response_matrix_filepath = None

        ### Set by run_gpr ###
        self.gpr_results : gpr.FitResults = None
        self.gpr_sigcontam_corrs : list[float] = None

        ### Set by save_rebinned_histograms ###
        self.rebinned_hists_filepath = None

        ### Set by run_plu ###
        self.plu_ws_filepath = None


##########################################################################################
###                                         RUN                                        ###
##########################################################################################


def save_rebinned_histograms(config : ChannelConfig):
    '''
    Rebins the CxAODReader histograms to match the response matrix/gpr.
    '''
    os.makedirs(f'{config.output_dir}/rebin', exist_ok=True)
    config.rebinned_hists_filepath = f'{config.output_dir}/rebin/{config.lepton_channel}lep_{{sample}}_rebin.root'

    output_files = {}
    def file(sample):
        if sample not in output_files:
            output_files[sample] = ROOT.TFile(config.rebinned_hists_filepath.format(sample=sample), 'RECREATE')
        return output_files[sample]
    
    variations = [utils.variation_nom] + utils.variations_hist
    for variable in config.variables:
        bins = utils.get_bins(config.lepton_channel, variable)
        bins = np.array(bins, dtype=float)
        for variation in variations:
            hist_name = '{sample}_VV{lep}Lep_MergHP_Inclusive_SR_'
            hist_name += utils.generic_var_to_lep(variable, config.lepton_channel).name
            hist_name = utils.hist_name_variation(hist_name, variation)

            hists = config.file_manager.get_hist_all_samples(config.lepton_channel, hist_name)
            for sample,hist in hists.items():
                hist = plot.rebin(hist, bins)
                hist.SetName(hist_name.format(lep=config.lepton_channel, sample=sample))

                f = file(sample)
                f.cd()
                hist.Write()
    
    plot.success(f'Saved rebinned histograms to {config.rebinned_hists_filepath}')


def run_gpr(channel_config : ChannelConfig, var : utils.Variable):
    '''
    Runs the GPR fit for every variation.
    '''
    plot.notice(f'{channel_config.log_base} running GPR fits for {var}')
    
    ### Config ###
    config_base = {
        'lepton_channel': channel_config.lepton_channel,
        'var': var,
        'output_dir': channel_config.output_dir + '/gpr',
        'use_vjets_mc': channel_config.is_asimov,
    }
    def run(config):
        gpr.run(channel_config.file_manager, config, channel_config.skip_fits or channel_config.skip_gpr)
    
    ### Nominal ###
    fit_config = gpr.FitConfig(
        variation='nominal',
        mu_stop=channel_config.mu_stop[0],
        mu_ttbar=channel_config.ttbar_fitter.mu_ttbar_nom[0],
        **config_base,
    )
    run(fit_config)

    ### Diboson signal strength variations ###
    mu_diboson_points = [0.9, 0.95, 1.05, 1.1]
    for mu_diboson in mu_diboson_points:
        fit_config = gpr.FitConfig(
            variation=f'mu-diboson{mu_diboson}',
            mu_stop=channel_config.mu_stop[0],
            mu_ttbar=channel_config.ttbar_fitter.mu_ttbar_nom[0],
            **config_base,
        )
        run(fit_config)
    channel_config.gpr_sigcontam_corrs = plot_gpr_mu_diboson_correlations(
        config=fit_config, 
        yields=mu_diboson_points,
        filename=f'{channel_config.output_dir}/plots/{fit_config.lepton_channel}lep_{fit_config.var}.gpr_diboson_mu_scan',
    )

    ### ttbar signal strength variations ###
    for variation in ['mu-ttbar_up', 'mu-ttbar_down']:
        fit_config = gpr.FitConfig(
            variation=variation,
            mu_stop=channel_config.mu_stop[0],
            mu_ttbar=channel_config.ttbar_fitter.get_var(variation),
            **config_base,
        )
        run(fit_config)

    ### Single top signal strength variations ###
    fit_config = gpr.FitConfig(
        variation='mu-stop_up',
        mu_stop=channel_config.mu_stop[0] + channel_config.mu_stop[1],
        mu_ttbar=channel_config.ttbar_fitter.get_var('mu-stop_up'),
        **config_base,
    )
    run(fit_config)
    fit_config = gpr.FitConfig(
        variation=f'mu-stop_down',
        mu_stop=channel_config.mu_stop[0] - channel_config.mu_stop[1],
        mu_ttbar=channel_config.ttbar_fitter.get_var('mu-stop_down'),
        **config_base,
    )
    run(fit_config)
    plot_gpr_ttbar_and_stop_correlations(fit_config, f'{channel_config.output_dir}/plots/{fit_config.lepton_channel}lep_{fit_config.var}.gpr_ttbar_stop_mu_scan')

    # TODO syst variations

    channel_config.gpr_results = fit_config.fit_results
    

def run_direct_fit(config : ChannelConfig, var : utils.Variable):
    '''
    Test fit where we fit the detector-level diboson signal strength to each bin directly.

    This will be superceded by the profile likelihood fit, but is a nice check.
    '''
    bins = utils.get_bins(config.lepton_channel, var)
    h_diboson_fit = ROOT.TH1F('h_diboson', 'Diboson', len(bins) - 1, np.array(bins, dtype=float))
    h_diboson_mc = ROOT.TH1F('h_diboson', 'Diboson', len(bins) - 1, np.array(bins, dtype=float))
    for i in range(len(bins) - 1):
        res = diboson_fit.run_fit(
            config=config,
            variable=var,
            bin=(bins[i], bins[i+1]),
            gpr_mu_corr=config.gpr_sigcontam_corrs[i],
        )
        diboson_yield = res['diboson-yield']
        h_diboson_fit.SetBinContent(i+1, diboson_yield[0])
        h_diboson_fit.SetBinError(i+1, diboson_yield[1])

        diboson_yield = res['diboson-yield-mc-statonly']
        h_diboson_mc.SetBinContent(i+1, diboson_yield[0])
        h_diboson_mc.SetBinError(i+1, diboson_yield[1])
    plot_yield_comparison(
        h_fit=h_diboson_fit, 
        h_mc=h_diboson_mc,
        filename=f'{config.output_dir}/plots/{config.lepton_channel}lep_{var}.directfit_yields',
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel',
            'Direct bin-by-bin fit',
        ],
        xtitle=f'{var:title}',
    )


def run_npcheck_drawfit(config : ChannelConfig, var : utils.Variable):
    '''
    Runs drawPostFit.C from NPCheck. This is supplanted by [plot_plu_fit].

    Uses the default fccs file in the NPCheck dir, so must call in sync with the fit!
    '''
    plot.notice(f'{config.log_base} drawing PLU fits using NPCheck')
    with open(f'{config.output_dir}/rf/log.{config.lepton_channel}lep_{var}.draw_fit.txt', 'w') as f:
        res = subprocess.run(
            ['./runDrawFit.py', config.plu_ws_filepath, 
                '--mu', '1', 
                '--fccs', 'fccs/FitCrossChecks.root'
            ], 
            cwd=config.npcheck_dir,
            stdout=f,
            stderr=f,
            # capture_output=True,
            # text=True,
        )
    res.check_returncode()
    npcheck_output_path = f'{config.npcheck_dir}/Plots/PostFit/summary_postfit_doAsimov0_doCondtional0_mu1.pdf'
    target_path = f'{config.output_dir}/rf/{config.lepton_channel}lep_{var}.plu_postfit.pdf'
    shutil.copyfile(npcheck_output_path, target_path)
    

def run_plu(config : ChannelConfig, var : utils.Variable):
    '''
    Runs the profile likelihood unfolding fit using ResonanceFinder. Assumes the
    GPR/response matrices have been created already.
    '''
    # Really important this import is here otherwise segfaults occur, due to the
    # `from ROOT import RF` line I think. But somehow hiding it here is fine.
    import rf_plu 

    os.makedirs(f'{config.output_dir}/rf', exist_ok=True)

    ### Create RF workspace ###
    if not config.skip_fits:
        plot.notice(f'{config.log_base} creating ResonanceFinder workspace')
        ws_path = rf_plu.run(
            lepton_channel=config.lepton_channel,
            variable=var,
            response_matrix_path=config.response_matrix_filepath,
            output_dir=config.output_dir,
            hist_file_format=config.rebinned_hists_filepath,
            mu_stop=config.mu_stop,
            mu_ttbar=config.ttbar_fitter.mu_ttbar_nom,
        )
    else:
        ws_path = rf_plu.ws_path(config.output_dir, config.lepton_channel, var)
    config.plu_ws_filepath = os.path.abspath(ws_path)

    ### Run fits ###
    fcc_path = f'{config.output_dir}/rf/{config.lepton_channel}lep_{var}.fcc.root'
    if not config.skip_fits:
        plot.notice(f'{config.log_base} running PLU fits')
        with open(f'{config.output_dir}/rf/log.{config.lepton_channel}lep_{var}.fcc.txt', 'w') as f:
            res = subprocess.run(
                ['./runFitCrossCheck.py', config.plu_ws_filepath],  # the './' is necessary!
                cwd=config.npcheck_dir, 
                stdout=f, 
                stderr=f
            ) 
        res.check_returncode()
        shutil.copyfile(f'{config.npcheck_dir}/fccs/FitCrossChecks.root', fcc_path)
    
    ### Get fit result ###
    fcc_file = ROOT.TFile(fcc_path)
    results_name = 'PlotsAfterGlobalFit/unconditionnal/fitResult'
    roofit_results = fcc_file.Get(results_name)
    roofit_results.Print()

    plu_fit_results = { v.GetName() : (v.getValV(), v.getError()) for v in roofit_results.floatParsFinal() }
    plu_fit_results['mu-stop'] = convert_alpha(plu_fit_results['alpha_mu-stop'], config.mu_stop)
    plu_fit_results['mu-ttbar'] = convert_alpha(plu_fit_results['alpha_mu-ttbar'], config.ttbar_fitter.mu_ttbar_nom)
    
    ### Draw fit ###
    plot_plu_fit(config, var, plu_fit_results)

    ### Draw yield vs MC ###
    bins = np.array(utils.get_bins(config.lepton_channel, var), dtype=float)
    h_fit = ROOT.TH1F('h_fit', '', len(bins) - 1, bins)
    for i in range(1, len(bins)):
        mu = plu_fit_results[f'mu_{i:02}']
        h_fit.SetBinContent(i, mu[0])
        h_fit.SetBinError(i, mu[1])
    h_mc = config.file_manager.get_hist(config.lepton_channel, utils.Sample.diboson, '{sample}_VV{lep}Lep_Merg_unfoldingMtx_' + var.name)
    h_mc = h_mc.ProjectionX()
    h_mc = plot.rebin(h_mc, bins)
    plot_yield_comparison(
        h_fit=h_fit, 
        h_mc=h_mc,
        filename=f'{config.output_dir}/plots/{config.lepton_channel}lep_{var}.plu_yields',
        text_pos='topright',
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel fiducial region',
        ],
        xtitle=f'{var:title}',
    )

    ### Draw pulls ###
    plot_pulls(config, var, plu_fit_results, f'{config.output_dir}/plots/{config.lepton_channel}lep_{var}.plu_pulls')

    ### Draw correlation matrix ###
    plot_correlations(config, var, roofit_results, f'{config.output_dir}/plots/{config.lepton_channel}lep_{var}.plu_corr')


def run_channel(config : ChannelConfig):
    '''
    Main run function for a single lepton channel.
    '''
    ### Generate response matricies ###
    plot.notice(f'{config.log_base} creating response matrix')
    config.response_matrix_filepath = unfolding.main(
        file_manager=config.file_manager,
        sample=utils.Sample.diboson,
        lepton_channel=config.lepton_channel,
        output=f'{config.output_dir}/response_matrix',
        vars=config.variables,
    )

    ### Rebin reco histograms ###
    save_rebinned_histograms(config)

    ### Iterate per variable ###
    for var in config.variables:
        ### GPR fit ###
        run_gpr(config, var)

        ### Diboson yield ###
        if not config.skip_fits:
            run_direct_fit(config, var)
    
        ### Prefit plot (pre-PLU but using GPR) ###
        plot_pre_plu_fit(config, var)

        ### Profile likelihood unfolding fit ###
        try: # This requires ResonanceFinder!
            run_plu(config, var)
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
    parser.add_argument('--skip-fits', action='store_true', help="Don't do the GPR or PLU fits. For the former, uses the fit results stored in the CSV. This file should be placed at '{output}/gpr/gpr_fit_results.csv'.")
    parser.add_argument('--skip-gpr', action='store_true', help="Skip only the GPR fits; uses the fit results stored in the CSV. This file should be placed at '{output}/gpr/gpr_fit_results.csv'.")
    parser.add_argument('--asimov', action='store_true', help="Use asimov data instead. Will look for files using [data-asimov] as the key instead of [data]. Create asimov data easily using make_asimov.py")
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
    if args.asimov:
        utils.Sample.data.file_stubs = ['data-asimov']
        args.output += '/asimov'

    file_manager = get_files(args.filepaths)
    os.makedirs(f'{args.output}/plots', exist_ok=True)

    plot.save_transparent_png = False
    plot.file_formats = ['png', 'pdf']

    ### ttbar fitter ###
    mu_stop = (1, 0.2)
    ttbar_fitter = ttbar_fit.TtbarSysFitter(file_manager, mu_stop_0=mu_stop)

    ### Loop over all channels ###
    for lepton_channel in [0]:
        config = ChannelConfig(
            lepton_channel=lepton_channel,
            file_manager=file_manager,
            ttbar_fitter=ttbar_fitter,
            mu_stop=mu_stop, 
            output_dir=args.output,
            skip_fits=args.skip_fits,
            skip_gpr=args.skip_gpr,
            is_asimov=args.asimov,
        )
        run_channel(config)
    


if __name__ == "__main__":
    main()