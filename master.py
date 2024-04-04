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
be replaced with the lepton channel name, and 'sample', which uses
[utils.Sample.file_stubs]. For example,

    hists/{lep}/{sample}.root

See [utils.FileManager] for details.
'''

from __future__ import annotations

import ROOT # type: ignore

import numpy as np
import os
import sys
import gc
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


class CondorSubmitMaker:
    '''
    Creates the condor.submit file for launching GPR jobs on HT Condor batch systems.

    Usage:
        maker = CondorSubmitMaker(config, var, 'submit.condor')
        maker.add(gpr_config1)
        maker.add(gpr_config2)
        ...
        maker.close()
    '''
    def __init__(self, config : ChannelConfig, var : utils.Variable, filepath : str):
        self.config = config
        self.var = var
        self.filepath = filepath
        if not os.path.exists(f'{config.output_dir}/condor_logs'):
            os.makedirs(f'{config.output_dir}/condor_logs')

        self.f = open(filepath, 'w')
        self.f.write(f'''\
# Setup
Universe = vanilla
getenv = True

# Log paths
output = {config.output_dir}/condor_logs/$(ClusterId).$(ProcId).out
error = {config.output_dir}/condor_logs/$(ClusterId).$(ProcId).err
log = {config.output_dir}/condor_logs/$(ClusterId).$(ProcId).log

# Queue
+JobFlavour = "10 minutes"
+queue="short"

# Command
Executable = gpr.py
transfer_input_files = plotting,utils.py

# Queue
queue arguments from (
''')

    def add(self, fit_config : gpr.FitConfig):
        input_paths = [os.path.abspath(x) for x in self.config.file_manager.file_path_formats]
        output_path = os.path.abspath(fit_config.output_plots_dir)
        # Output hists and plots to same finely binned directory so no overwrite between jobs!

        self.f.write('    ')
        self.f.write(' '.join(input_paths))
        self.f.write(f' --lepton {self.config.lepton_channel}')
        self.f.write(f' --var {self.var}')
        self.f.write(f' --output {output_path}')
        if self.config.is_asimov:
            self.f.write(f' --closure-test') 
        self.f.write(f' --variation {fit_config.variation}')
        self.f.write(f' --mu-ttbar {fit_config.mu_ttbar}')
        self.f.write(f' --mu-stop {fit_config.mu_stop}\n')
    
    def close(self):
        self.f.write(')')
        self.f.close()
        plot.success(f'Wrote condor submit file to {self.filepath}')

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
    variations = ['nominal'] + [x + y for x in [utils.variation_mu_ttbar, utils.variation_mu_stop] for y in [utils.variation_up_key, utils.variation_down_key]]
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


def plot_yield_comparison(h_fit, h_mc, h_eft=None, eft_legend=None, **plot_opts):
    '''
    Plots a 1D comparison between a fit and MC. The fit is shown as data points while the
    MC is shown as a filled blue band. Includes a fit/MC ratio.
    '''
    h_mc.SetMarkerSize(0)
    h_mc.SetLineColor(plot.colors.blue)
    h_mc_err = h_mc.Clone()
    h_mc_err.SetFillColorAlpha(plot.colors.blue, 0.3)

    if h_eft:
        h_eft.SetMarkerSize(0)
        h_eft.SetLineColor(plot.colors.orange)
        h_eft_err = h_eft.Clone()
        h_eft_err.SetFillColorAlpha(plot.colors.orange, 0.3)

    ratio = h_fit.Clone()
    ratio.Divide(h_mc)

    if h_eft:
        objs = [h_mc, h_mc_err, h_eft, h_eft_err, h_fit]
        legend = ['', 'SM', '', f'SM + {eft_legend}', 'Fit']
        opts = ['HIST', 'E2', 'HIST', 'E2', 'PE']
        legend_opts = ['', 'FL', '', 'FL', 'PEL']
    else:
        objs = [h_mc, h_mc_err, h_fit]
        legend = ['', 'MC', 'Fit']
        opts = ['HIST', 'E2', 'PE']
        legend_opts = ['', 'FL', 'PEL']

    ROOT.gStyle.SetErrorX(0)
    plot.plot_ratio(
        objs1=objs,
        objs2=[ratio],
        legend=legend,
        opts=opts,
        legend_opts=legend_opts,
        opts2='PE',
        ytitle='Events',
        ytitle2='Fit / SM' if h_eft else 'Fit / MC',
        hline=1,
        y_range2=(0.5, 1.5),
        logy=True,
        **plot_opts,
    )
    ROOT.gStyle.SetErrorX()


def plot_plu_yields(config : ChannelConfig, variable : utils.Variable, plu_fit_results, filename : str):
    '''
    Uses [plot_yield_comparison] to plot the PLU unfolded result against the fiducial MC.
    '''
    bins = np.array(utils.get_bins(config.lepton_channel, variable), dtype=float)

    ### Fit ###
    h_fit = ROOT.TH1F('h_fit', '', len(bins) - 1, bins)
    for i in range(1, len(bins)):
        mu = plu_fit_results[f'mu_{i:02}']
        h_fit.SetBinContent(i, mu[0])
        h_fit.SetBinError(i, mu[1])

    ### MC ###
    # Buggy response matrix in eos_v3 files, so point to local files in interim
    # Note fixed in https://gitlab.cern.ch/CxAODFramework/CxAODReader_VVSemileptonic/-/merge_requests/445
    temp_file_manager = utils.FileManager( 
        samples=[utils.Sample.diboson, utils.Sample.cw_lin, utils.Sample.cw_quad],
        file_path_formats=['../../{lep}/{lep}_{sample}_x_Feb24-ANN.hists.root'],
        lepton_channels=[0, 1, 2],
    )
    def get(sample):
        # h = config.file_manager.get_hist(config.lepton_channel, sample, '{sample}_VV{lep}_Merg_unfoldingMtx_' + variable.name)
        h = temp_file_manager.get_hist(config.lepton_channel, sample, '{sample}_VV{lep}_Merg_unfoldingMtx_' + variable.name)
        return plot.rebin(h.ProjectionX(), bins)
    h_mc = get(utils.Sample.diboson)

    ### EFT ###
    cw = 0.12
    h_cw_quad = get(utils.Sample.cw_quad)
    h_cw_lin = get(utils.Sample.cw_lin)
    h_cw_quad.Scale(cw**2)
    h_cw_lin.Scale(cw)
    h_cw_quad.Add(h_cw_lin)
    h_cw_quad.Add(h_mc)

    ### Plot ###
    yield_args = dict(
        h_fit=h_fit,
        h_mc=h_mc,
        text_pos='topright',
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel fiducial region',
            'MC errors are stat only',
        ],
        xtitle=f'{variable:title}',
    )
    plot_yield_comparison(**yield_args, filename=filename)
    plot_yield_comparison(**yield_args, h_eft=h_cw_quad, eft_legend='c_{W}^{quad}=' + f'{cw:.2f}', filename=filename + '_cw')


def plot_pre_plu_fit(config : ChannelConfig, variable : utils.Variable):
    '''
    Plots a stack plot comparing data to backgrounds prior to the PLU fit but using the
    GPR background estimate.
    '''
    ### Get hists ###
    f_gpr = ROOT.TFile(f'{config.output_dir}/gpr/gpr_{config.lepton_channel}lep_vjets_yield.root')
    h_gpr = f_gpr.Get('Vjets_SR_' + variable.name)

    hist_name = '{sample}_VV{lep}_MergHP_Inclusive_SR_' + utils.generic_var_to_lep(variable, config.lepton_channel).name
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

    ### Sum and ratios ###
    h_sum = h_gpr.Clone()
    h_sum.Add(h_diboson)
    h_sum.Add(h_ttbar)
    h_sum.Add(h_stop)

    h_ratio = h_data.Clone()
    h_errs = h_sum.Clone()
    h_ratio.Divide(h_sum)
    h_errs.Divide(h_sum)

    stack = [
        (h_diboson.Integral(), h_diboson, 'Diboson (#mu=1)', plot.colors.pastel_blue),
        (h_stop.Integral(), h_stop, 'Single top', plot.colors.pastel_yellow),
        (h_ttbar.Integral(), h_ttbar, 't#bar{t}', plot.colors.pastel_orange),
        (h_gpr.Integral(), h_gpr, 'GPR (V+jets)', plot.colors.pastel_red),
    ]
    stack.sort()

    ### Plot ###
    pads = plot.RatioPads()
    plotter1 = pads.make_plotter1(
        ytitle='Events',
        logy=True,
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel pre-PLU',
        ],
        y_min=0.2,
    )
    plotter1.add(
        objs=[x[1] for x in stack],
        legend=[x[2] for x in stack],
        stack=True,
        opts='HIST',
        fillcolor=[x[3] for x in stack],
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
    hist_name = '{sample}_VV{lep}_MergHP_Inclusive_SR_' + utils.generic_var_to_lep(variable, config.lepton_channel).name
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

    stack = [
        (h_stop.Integral(), h_stop, 'top', plot.colors.pastel_yellow),
        (h_ttbar.Integral(), h_ttbar, 't#bar{t}', plot.colors.pastel_orange),
        (h_gpr.Integral(), h_gpr, 'V+jets', plot.colors.pastel_red),
    ]
    stack.sort()

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
    bkg_colors = [x[3] for x in stack]
    def stack_color(i):
        if i < len(h_signals):
            return palette_colors[i].GetNumber()
        else:
            return bkg_colors[i - len(h_signals)]

    ### Legend ###
    legend = [f'Bin {i}' for i in range(1, len(bins))][::-1]
    legend += [x[2] for x in stack]

    ### Plot stack ###
    plotter1.add(
        objs=h_signals[::-1] + [x[1] for x in stack],
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
        legend=['#splitline{GPR +}{MC stat}'],
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
    ### Get parameter lists ###
    # bins = utils.get_bins(config.lepton_channel, variable)
    # alphas = [utils.variation_lumi] + utils.variations_custom + utils.variations_hist
    # mus = [f'mu_{x:02}' for x in range(1, len(bins))]
    # pars = [f'alpha_{x}' for x in alphas] + mus

    ### All pars (custom sort) ###
    def sort_name(name):
        sname = name.replace('alpha_', '').replace('gamma_', '')
        if 'mu' in sname:
            sname = '0' + sname
        elif 'stat' in sname:
            sname = '1' + sname
        elif 'lumiNP' == sname:
            sname = '2' + sname
        return (sname, name)
    pars_all = [ sort_name(v.GetName()) for v in roofit_results.floatParsFinal() ]
    pars_all.sort()
    pars_all = [ x[1] for x in pars_all ]

    ### Pruned pars ###
    prune_threshold = 0.2
    pars_prune = []
    for i,par in enumerate(pars_all):
        if 'mu' in par:
            pars_prune.append(par)
        else:
            for j in range(len(pars_all)):
                if j == i: continue
                if abs(roofit_results.correlation(pars_all[i], pars_all[j])) > prune_threshold:
                    pars_prune.append(par)
                    break

    ### Create matrices ###
    def create_hist(pars):
        n = len(pars)
        h = ROOT.TH2F('h_cov', '', n, 0, n, n, 0, n)
        for i in range(n):
            par_label = pars[i].replace('alpha_', '').replace('gamma_', '')
            h.GetXaxis().SetBinLabel(i + 1, par_label)
            h.GetYaxis().SetBinLabel(n - i, par_label)
            for j in range(n):
                h.SetBinContent(i + 1, n - j, roofit_results.correlation(pars[i], pars[j]))
        h.GetXaxis().LabelsOption('v')
        return h
    h_all = create_hist(pars_all)
    h_prune = create_hist(pars_prune)

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
    common_args = dict(
        text_pos='topright',
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel {variable} fit',
        ],
        opts='COLZ',
        z_range=[-1, 1],
        left_margin=0.3,
        bottom_margin=0.3,
    )
    plot.plot(
        objs=[h_all],
        filename=filename,
        **common_args,
    )
    plot.plot(
        objs=[h_prune],
        filename=filename + f'_prune{prune_threshold}',
        **common_args,
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
            skip_hist_gen : bool,
            skip_fits : bool,
            skip_direct_fit : bool,
            skip_gpr : bool,
            skip_gpr_if_present : bool,
            gpr_condor : bool,
            is_asimov : bool,
            run_plu_val : bool,
        ):
        self.lepton_channel = lepton_channel
        self.file_manager = file_manager
        self.ttbar_fitter = ttbar_fitter
        self.mu_stop = mu_stop
        self.output_dir = output_dir
        self.skip_hist_gen = skip_hist_gen
        self.skip_fits = skip_fits
        self.skip_direct_fit = skip_direct_fit
        self.skip_gpr = skip_gpr
        self.skip_gpr_if_present = skip_gpr_if_present
        self.gpr_condor = gpr_condor
        self.is_asimov = is_asimov
        self.plu_validation_iters = 100 if run_plu_val else 0

        self.npcheck_dir = 'ResonanceFinder/NPCheck'
        self.log_base = f'master.py::run_channel({lepton_channel}lep)'
        if lepton_channel == 0:
            self.variables = [utils.Variable.vv_mt]
        elif lepton_channel == 1:
            self.variables = [utils.Variable.vv_m]
        elif lepton_channel == 2:
            self.variables = [utils.Variable.vv_m]

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

    ### Output ###
    output_files = {}
    def file(sample):
        if sample not in output_files:
            output_files[sample] = ROOT.TFile(config.rebinned_hists_filepath.format(sample=sample), 'RECREATE')
        return output_files[sample]

    ### All histo variations ###
    variations = [utils.variation_nom]
    for x in utils.variations_hist:
        variations.append(x + utils.variation_up_key)
        variations.append(x + utils.variation_down_key)

    ### Loop per variable ###
    for variable in config.variables:
        bins = utils.get_bins(config.lepton_channel, variable)
        bins = np.array(bins, dtype=float)
        for variation in variations:
            hist_name = '{sample}_VV{lep}_MergHP_Inclusive_SR_'
            hist_name += utils.generic_var_to_lep(variable, config.lepton_channel).name

            hists = config.file_manager.get_hist_all_samples(config.lepton_channel, hist_name, variation)
            for sample_name,hist in hists.items():
                if sample_name == utils.Sample.data.name: continue # Handled in [save_data_variation_histograms]

                hist = plot.rebin(hist, bins)
                new_name = hist_name.format(lep=f'{config.lepton_channel}Lep', sample=sample_name)
                new_name = utils.hist_name_variation(new_name, config.file_manager.samples[sample_name], variation, separator='__')
                # For some reason RF expects a double underscore...not sure how to change
                hist.SetName(new_name)

                f = file(sample_name)
                f.cd()
                hist.Write()

    ### Data rebinned histograms for PLU stat test ###
    save_data_variation_histograms(config, file(utils.Sample.data.name))

    plot.success(f'Saved rebinned histograms to {config.rebinned_hists_filepath}')


def save_data_variation_histograms(config : ChannelConfig, f : ROOT.TFile):
    '''
    Saves several statistical variations of the data histograms
    '''
    ### Loop per variable ###
    for variable in config.variables:
        ### Config ###
        bins = utils.get_bins(config.lepton_channel, variable)
        bins = np.array(bins, dtype=float)

        hist_name = '{sample}_VV{lep}_MergHP_Inclusive_SR_'
        hist_name += utils.generic_var_to_lep(variable, config.lepton_channel).name

        ### Nominal hist ###
        h_nom = config.file_manager.get_hist(config.lepton_channel, utils.Sample.data, hist_name)
        h_nom = plot.rebin(h_nom, bins)
        h_nom.SetName(hist_name.format(lep=f'{config.lepton_channel}Lep', sample='data'))

        f.cd()
        h_nom.Write()

        ### Create variations ###
        rng = np.random.default_rng()
        for i in range(config.plu_validation_iters):
            name = hist_name.format(lep=f'{config.lepton_channel}Lep', sample=f'data_var{i:03}')
            h = ROOT.TH1F(name, name, len(bins) - 1, bins)
            for x in range(1, h_nom.GetNbinsX() + 1):
                h[x] = rng.poisson(h_nom[x])
            h.Write()


def run_gpr(channel_config : ChannelConfig, var : utils.Variable):
    '''
    Runs the GPR fit for every variation.
    '''
    plot.notice(f'{channel_config.log_base} running GPR fits for {var}')

    ### Condor ###
    if channel_config.gpr_condor:
        condor_file = CondorSubmitMaker(channel_config, var, f'{channel_config.output_dir}/gpr/{channel_config.lepton_channel}lep_{var}.submit.condor')

    ### Config ###
    def make_config(variation, mu_stop):
        return gpr.FitConfig(
            lepton_channel=channel_config.lepton_channel,
            var=var,
            output_hists_dir=f'{channel_config.output_dir}/gpr',
            output_plots_dir=f'{channel_config.output_dir}/gpr/{channel_config.lepton_channel}lep/{var}/{variation}',
            use_vjets_mc=channel_config.is_asimov,
            variation=variation,
            mu_ttbar=channel_config.ttbar_fitter.get_var(variation),
            mu_stop=mu_stop,
        )
    mu_diboson_points = [0.9, 0.95, 1.05, 1.1]
    
    ### Summary plots ###
    def summary_actions():
        fit_config = make_config(utils.variation_nom, channel_config.mu_stop)
        channel_config.gpr_results = fit_config.fit_results
        channel_config.gpr_sigcontam_corrs = plot_gpr_mu_diboson_correlations(
            config=fit_config,
            yields=mu_diboson_points,
            filename=f'{channel_config.output_dir}/plots/{fit_config.lepton_channel}lep_{fit_config.var}.gpr_diboson_mu_scan',
        )
        plot_gpr_ttbar_and_stop_correlations(fit_config, f'{channel_config.output_dir}/plots/{fit_config.lepton_channel}lep_{fit_config.var}.gpr_ttbar_stop_mu_scan')
    
    if channel_config.skip_fits or channel_config.skip_gpr:
        summary_actions()
        return

    ### Common run function ###
    def run(variation, mu_stop=channel_config.mu_stop[0]):
        config = make_config(variation, mu_stop)
        if channel_config.gpr_condor:
            condor_file.add(config)
        else:
            gpr.run(
                file_manager=channel_config.file_manager, 
                config=config, 
                from_csv_only=channel_config.skip_fits or channel_config.skip_gpr,
                skip_if_in_csv=channel_config.skip_gpr_if_present,
            )

    ### Nominal ###
    run('nominal')

    ### Diboson signal strength variations ###
    for mu_diboson in mu_diboson_points:
        run(f'mu-diboson{mu_diboson}')

    ### ttbar signal strength variations ###
    for updown in [utils.variation_up_key, utils.variation_down_key]:
        run(utils.variation_mu_ttbar + updown)

    ### Single top signal strength variations ###
    run(
        variation=utils.variation_mu_stop + utils.variation_up_key,
        mu_stop=channel_config.mu_stop[0] + channel_config.mu_stop[1],
    )
    run(
        variation=utils.variation_mu_stop + utils.variation_down_key,
        mu_stop=channel_config.mu_stop[0] - channel_config.mu_stop[1],
    )

    ### Syst variations ###
    for variation_base in utils.variations_hist:
        for updown in [utils.variation_up_key, utils.variation_down_key]:
            run(variation_base + updown)

    ### Outputs ###
    if channel_config.gpr_condor:
        condor_file.close()
        res = subprocess.run(['condor_submit', condor_file.filepath])
        if res.returncode == 0:
            plot.success("Launched GPR jobs on condor. Once jobs are done, merge the results using merge_gpr_condor.py, then recall master.py using --skip-gpr.")
        else:
            plot.error(f"Couldn't launch condor jobs: {res}.")
    else:
        summary_actions()


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


def run_plu(config : ChannelConfig, var : utils.Variable, stat_validation_index : int = None):
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
            stat_validation_index=stat_validation_index,
        )
    else:
        ws_path = rf_plu.ws_path(config.output_dir, config.lepton_channel, var, stat_validation_index)
    ws_path = os.path.abspath(ws_path)
    if stat_validation_index is None:
        config.plu_ws_filepath = ws_path

    ### Run fits ###
    if stat_validation_index is None:
        fcc_path = f'{config.output_dir}/rf/{config.lepton_channel}lep_{var}.fcc.root'
    else:
        fcc_path = f'{config.output_dir}/rf/{config.lepton_channel}lep_{var}.fcc_var{stat_validation_index:03}.root'
    if not config.skip_fits:
        plot.notice(f'{config.log_base} running PLU fits')
        with open(f'{config.output_dir}/rf/log.{config.lepton_channel}lep_{var}.fcc.txt', 'w') as f:
            res = subprocess.run(
                ['./runFitCrossCheck.py', ws_path],  # the './' is necessary!
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
    if not config.skip_fits:
        roofit_results.Print()

    plu_fit_results = { v.GetName() : (v.getValV(), v.getError()) for v in roofit_results.floatParsFinal() }
    plu_fit_results['mu-stop'] = convert_alpha(plu_fit_results['alpha_mu-stop'], config.mu_stop)
    plu_fit_results['mu-ttbar'] = convert_alpha(plu_fit_results['alpha_mu-ttbar'], config.ttbar_fitter.mu_ttbar_nom)

    ### Plots ###
    if stat_validation_index is None:
        ### Draw fit ###
        plot_plu_fit(config, var, plu_fit_results)

        ### Draw pulls ###
        plot_pulls(config, var, plu_fit_results, f'{config.output_dir}/plots/{config.lepton_channel}lep_{var}.plu_pulls')

        ### Draw correlation matrix ###
        plot_correlations(config, var, roofit_results, f'{config.output_dir}/plots/{config.lepton_channel}lep_{var}.plu_corr')

        ### Draw yield vs MC ###
        plot_plu_yields(config, var, plu_fit_results, f'{config.output_dir}/plots/{config.lepton_channel}lep_{var}.plu_yields')

    return plu_fit_results


def run_plu_val(config : ChannelConfig, variable : utils.Variable, results_nom : dict[str, tuple[float, float]]):
    '''
    Runs a validation test for the PLU fit by creating statistical variations of the data
    histograms (see [save_data_variation_histograms]) and running the PLU on each of them,
    generating a distribution of the output results.
    '''
    bins = utils.get_bins(config.lepton_channel, variable)
    nbins = len(bins) - 1

    ### Create histograms ###
    hists = []
    for i in range(nbins):
        hists.append(ROOT.TH1F(f'h_plu_validation_{i}', '', 20, -5, 5))
    sums = np.zeros(nbins)
    sum_squares = np.zeros(nbins)

    ### Run PLU multiple times ###
    for val_index in range(config.plu_validation_iters):
        results = run_plu(config, variable, val_index)
        for i in range(nbins):
            name = f'mu_{i + 1:02}'
            val_nom = results_nom[name]
            val = (results[name][0] - val_nom[0]) / val_nom[1]

            hists[i].Fill(val)
            sums[i] += val
            sum_squares[i] += val**2

    ### Mean and std dev callback ###
    means = sums / config.plu_validation_iters
    std_devs = np.sqrt(sum_squares / config.plu_validation_iters - means**2)
    def callback(plotter):
        for i in range(nbins):
            line = ROOT.TLine(means[i], i, means[i], i + 0.5)
            line.SetLineColor(ROOT.kBlack)
            line.SetLineWidth(2)
            line.Draw()

            line2 = ROOT.TLine(means[i] - std_devs[i], i, means[i] - std_devs[i], i + 0.5)
            line2.SetLineColor(ROOT.kBlack)
            line.SetLineWidth(2)
            line2.SetLineStyle(ROOT.kDashed)
            line2.Draw()

            line3 = ROOT.TLine(means[i] + std_devs[i], i, means[i] + std_devs[i], i + 0.5)
            line3.SetLineColor(ROOT.kBlack)
            line.SetLineWidth(2)
            line3.SetLineStyle(ROOT.kDashed)
            line3.Draw()

            plotter.cache.extend([line, line2, line3])

    ### Legend for mean lines ###
    line_mean = ROOT.TLine(0, 0, 1, 1)
    line_mean.SetLineColor(ROOT.kBlack)
    line_mean.SetLineWidth(2)

    line_var = ROOT.TLine(0, 0, 1, 1)
    line_var.SetLineColor(ROOT.kBlack)
    line_var.SetLineWidth(2)
    line_var.SetLineStyle(ROOT.kDashed)
    legend = [
        (line_mean, 'Mean', 'L'),
        (line_var, '#pm1 std. dev.', 'L'),
    ]

    ### Plot ###
    plot.plot_tiered(
        hists=[[h] for h in hists],
        text_pos='top',
        tier_labels=[f'Bin {i:02}' for i in range(1, len(bins))],
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel',
            'PLU fit results from Poisson varied data',
        ],
        legend=legend,
        xtitle='Fitted Fiducial Events #frac{x - x_{nom}}{#sigma_{nom}}',
        text_offset_top = 0.02,
        y_pad_top=0.2,
        title_offset_x=1.5,
        bottom_margin=0.2,
        fillcolor=plot.colors.pastel,
        callback=callback,
        filename=f'{config.output_dir}/plots/{config.lepton_channel}lep_{variable}.plu_validation',
    )


def run_channel(config : ChannelConfig):
    '''
    Main run function for a single lepton channel.
    '''
    if not config.skip_hist_gen and not config.skip_fits and not config.skip_gpr:
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
    else:
        config.response_matrix_filepath = unfolding.output_path(f'{config.output_dir}/response_matrix', utils.Sample.diboson, config.lepton_channel)
        config.rebinned_hists_filepath = f'{config.output_dir}/rebin/{config.lepton_channel}lep_{{sample}}_rebin.root'

    ### Iterate per variable ###
    for var in config.variables:
        ### GPR fit ###
        gc.collect() # Get segfaults when generating plots sometimes
        gc.disable() # https://root-forum.cern.ch/t/segfault-on-creating-canvases-and-pads-in-a-loop-with-pyroot/44729/13
        run_gpr(config, var) # When skip_gpr, still generates the summary plots
        gc.enable()
        if config.gpr_condor:
            continue

        ### Diboson yield ###
        if not config.skip_fits and not config.skip_direct_fit:
            run_direct_fit(config, var)

        ### Prefit plot (pre-PLU but using GPR) ###
        plot_pre_plu_fit(config, var)

        ### Profile likelihood unfolding fit ###
        try: # This requires ResonanceFinder!
            plu_results = run_plu(config, var)
            if config.plu_validation_iters:
                run_plu_val(config, var, plu_results)
        except Exception as e:
            plot.warning(str(e))
            raise e




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
    parser.add_argument('--skip-hist-gen', action='store_true', help="Skip generating the response matrix and rebinned histograms.")
    parser.add_argument('--skip-fits', action='store_true', help="Don't do the GPR or PLU fits. For the former, uses the fit results stored in the CSV. This file should be placed at '{output}/gpr/gpr_fit_results.csv'.")
    parser.add_argument('--skip-direct-fit', action='store_true', help="Don't do the direct diboson yield fit.")
    parser.add_argument('--skip-gpr', action='store_true', help="Skip only the GPR fits; uses the fit results stored in the CSV. This file should be placed at '{output}/gpr/gpr_fit_results.csv'.")
    parser.add_argument('--skip-gpr-if-present', action='store_true', help="Will run the GPR fits but skip the ones that are present in the CSV already.")
    parser.add_argument('--no-systs', action='store_true', help="Run without using the systematic variations.")
    parser.add_argument('--asimov', action='store_true', help="Use asimov data instead. Will look for files using [data-asimov] as the naming key instead of [data]. Create asimov data easily using make_asimov.py")
    parser.add_argument('--run-plu-val', action='store_true', help="Runs a PLU validation test by varying the data histogram and performing the entire PLU fit multiple times.")
    parser.add_argument('--condor', action='store_true', help="Runs the GPR fits via HT Condor. Merge the results using merge_gpr_condor.py, then recall master.py using --skip-gpr.")
    parser.add_argument('--mu-stop', default='1,0.2', help="The stop signal strength. Should be a comma-separated pair val,err.")
    parser.add_argument('--channels', default='0,1,2', help="The lepton channels to run over, separated by commas.")
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
            utils.Sample.cw_lin,
            utils.Sample.cw_quad,
        ],
        file_path_formats=filepaths,
    )
    return file_manager


def main():
    ### Args ###
    args = parse_args()
    if args.asimov:
        utils.Sample.data.file_stubs = ['data-asimov']
        args.output += '/asimov'
    if args.no_systs:
        utils.variations_hist.clear()
    mu_stop = [float(x) for x in args.mu_stop.split(',')]
    channels = [int(x) for x in args.channels.split(',')]

    ### Input files ###
    file_manager = get_files(args.filepaths)

    ### Output files ###
    os.makedirs(f'{args.output}/plots', exist_ok=True)
    plot.save_transparent_png = False
    plot.file_formats = ['png', 'pdf']

    ### ttbar fitter ###
    ttbar_fitter = ttbar_fit.TtbarSysFitter(file_manager, mu_stop_0=mu_stop)

    ### Loop over all channels ###
    for lepton_channel in channels:
        config = ChannelConfig(
            lepton_channel=lepton_channel,
            file_manager=file_manager,
            ttbar_fitter=ttbar_fitter,
            mu_stop=mu_stop,
            output_dir=args.output,
            skip_hist_gen=args.skip_hist_gen,
            skip_fits=args.skip_fits,
            skip_direct_fit=args.skip_direct_fit,
            skip_gpr=args.skip_gpr,
            skip_gpr_if_present=args.skip_gpr_if_present,
            gpr_condor=args.condor,
            is_asimov=args.asimov,
            run_plu_val=args.run_plu_val,
        )
        run_channel(config)


if __name__ == "__main__":
    main()
