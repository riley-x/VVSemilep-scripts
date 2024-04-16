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

Check [utils.Sample] to make sure the hardcoded naming structures are correct.

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------

    kinematics.py filepath/formatter_1.root [...]

This will fetch files using the naming convention supplied in the arguments. These
arguments can include python formatters (using curly braces) for 'lep', which will be
replaced with the lepton channel number, and 'sample', which uses
[utils.Sample.file_stubs]. For example,

    hists/{lep}/{sample}.root
    
See [utils.FileManager] for details.
'''

from plotting import plot
import ROOT # type: ignore
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


def _plot_hists_and_percentage(hists, **plot_opts):
    '''
    Plots each hist in 'PE' mode, unstacked, to compare their relative distributions 
    and sizes. A stacked ratio subplot shows their relative % contribution.
    '''
    ### Clean negative bins ###
    hists = [h.Clone() for h in hists]
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
    default_opts = dict(
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
        linewidth2=1,
        y_range2=[0, 100],
        outlier_arrows=False,
    )
    default_opts.update(plot_opts)
    return plot.plot_ratio(**default_opts)


def plot_MC_backgrounds_per_region(file_manager : utils.FileManager, output_dir : str):
    '''
    Plots each sample in 'PE' mode, unstacked, to compare their relative distributions 
    and sizes. A ratio subplot shows their relative contribution.

    Repeats for each inclusive HP region, each variable, and both a fine binning and the
    unfolding binning.
    '''
    vars : list[utils.Variable] = [
        utils.Variable.vv_m,
        utils.Variable.vv_mt,
    ]

    for lep in [0, 1, 2]:
        for region in [f'MergHP_Inclusive_{x}' for x in ['SR', 'TCR', 'MCR']]:
            if lep == 0 and 'TCR' in region: continue
            for var in vars:
                ### Get hists per sample ###
                hists = []
                legend = []
                for sample in file_manager.samples.values():
                    if 'data' in sample.name: continue
                    lep_var = utils.generic_var_to_lep(var, lep)
                    h = file_manager.get_hist(lep, sample.name, f'{{sample}}_VV{{lep}}_{region}_{lep_var}')
                    if h is not None:
                        hists.append(h)
                        legend.append(sample.title)
                if not hists: continue

                ### Plot opts ###
                opts = dict(
                    subtitle=[
                        '#sqrt{s}=13 TeV, 140 fb^{-1}',
                        f'{lep}Lep {region.replace("_", " ")}',
                    ],
                    legend=legend,
                    **var.x_plot_opts(),
                )

                ### Fine binning ###
                _plot_hists_and_percentage(
                    hists=[plot.rebin(h, var.rebin) for h in hists], 
                    filename=f'{output_dir}/mc_bkg.{lep}lep.{region}.{var}.fine',
                    **opts,
                )

                ### Unfolding binning ###
                bins = utils.get_bins(lep, var)
                _plot_hists_and_percentage(
                    hists=[plot.rebin(h, bins) for h in hists], 
                    filename=f'{output_dir}/mc_bkg.{lep}lep.{region}.{var}.coarse',
                    **opts,
                )


def plot_MC_backgrounds_for_fatjet_m(file_manager : utils.FileManager, output_dir : str):
    '''
    Plots each sample in 'PE' mode, unstacked, to compare their relative distributions 
    and sizes. A ratio subplot shows their relative contribution.

    Specialized for m(J) to plot both the SR and MCR, with the SR window outlined.
    '''
    var = utils.Variable.fatjet_m
    sr_window = (72, 102)

    def callback(plotter1, plotter2):
        '''Add the SR window'''
        plotter1.draw_vline(x=sr_window[0])
        plotter1.draw_vline(x=sr_window[1])
        plotter2.draw_vline(x=sr_window[0])
        plotter2.draw_vline(x=sr_window[1])

    for lep in [0, 1, 2]:
        hists = []
        legend = []
        for sample in file_manager.samples.values():
            if 'data' in sample.name: continue
            h_sr = file_manager.get_hist(lep, sample.name, f'{{sample}}_VV{{lep}}_MergHP_Inclusive_SR_{var}')
            h_cr = file_manager.get_hist(lep, sample.name, f'{{sample}}_VV{{lep}}_MergHP_Inclusive_MCR_{var}')
            if h_sr is None or h_cr is None: continue
            h_sr.Add(h_cr)
            h = plot.rebin(h_sr, var.rebin)
            hists.append(h)
            legend.append(sample.title)
        if not hists: continue

        _plot_hists_and_percentage(
            hists=hists, 
            filename=f'{output_dir}/mc_bkg_fatjet_m.{lep}lep',
            subtitle=[
                '#sqrt{s}=13 TeV, 140 fb^{-1}',
                f'{lep}Lep MergHP Inclusive SR/MCR',
            ],
            legend=legend,
            **var.x_plot_opts(),
            callback=callback,
        )


@run
def plot_MC_main_channel_yields(file_manager : utils.FileManager, output_dir : str):
    '''
    Plots a cutflow-esque graph that shows the MC yields in each of the main channels (lep
    x SR/CR). Includes a ratio plot that shows the percentage of total yield.
    '''
    ### Config ###
    sample_colors = {
        utils.Sample.wjets.name:   plot.colors.blue,
        utils.Sample.zjets.name:   plot.colors.purple,
        utils.Sample.ttbar.name:   plot.colors.orange,
        utils.Sample.stop.name:    plot.colors.red,
        utils.Sample.diboson.name: plot.colors.green,
    }

    ### Create "cutflow" hists ###
    hists = {}
    legend = []
    colors = []
    colors_40 = []
    for _,sample in file_manager.samples.items():
        if sample == utils.Sample.data: continue
        hists[sample.name] = ROOT.TH1F('h_{sample}', '', 8, 0, 8)
        legend.append(sample.title)
        colors.append(sample_colors[sample.name])
        colors_40.append(plot.colors.tableu_40_l[plot.colors.tableu_l.index(sample_colors[sample.name])])
    
    ### Fill each bin ###
    i_bin = 0
    bin_labels = []
    vertical_lines = []
    for lep in [0, 1, 2]:
        var = utils.Variable.vvJ_mt if lep == 0 else utils.Variable.lvJ_m if lep == 1 else utils.Variable.llJ_m
        for region in ['SR', 'MCR', 'TCR']:
            if lep == 0 and region == 'TCR': continue
            
            sample_hists = file_manager.get_hist_all_samples(lep, f'{{sample}}_VV{{lep}}_MergHP_Inclusive_{region}_{var}')
            for sample,h_cutflow in hists.items():
                v,e = plot.integral_user(sample_hists[sample], return_error=True)
                h_cutflow[i_bin + 1] = v
                h_cutflow.SetBinError(i_bin + 1, e)
            
            bin_labels.append(f'{lep}lep {region}')
            i_bin += 1
        vertical_lines.append(i_bin)
    
    ### Ratios ###
    hists_l = list(hists.values())
    h_sum = hists_l[0].Clone()
    for h in hists_l[1:]:
        h_sum.Add(h)

    ratios = []
    for h in hists_l:
        h = h.Clone()
        h.Divide(h_sum)
        h.Scale(100)
        ratios.append(h)

    ### Plot main ###
    pads = plot.RatioPads(height1=0.6, bottom_margin=0.16)
    plotter1 = pads.make_plotter1(
        objs=hists_l,
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            'VV Merged HP Inclusive Yields',
        ],
        legend=legend,
        ytitle='Events',
        x_bin_labels=bin_labels, # not shown, but matches the ticks with subplot
        linecolor=colors,
        markercolor=colors,
        logy=True,
    )
    for x in vertical_lines[:-1]:
        line = ROOT.TLine(x, plotter1.y_range[0], x, plotter1.data_y_max * 1.2)
        line.SetLineWidth(2)
        line.Draw()
        plotter1.cache.append(line)
    
    ### Plot ratio ###
    plotter2 = pads.make_plotter2(
        objs=ratios,
        ytitle='% of Total',
        xtitle='Region',
        x_bin_labels=bin_labels,
        x_labels_option='u',
        label_offset_x=0.015,
        title_offset_x=1.5,
        linewidth=1,
        linecolor=colors,
        fillcolor=colors_40,
        stack=True,
        opts='HIST',
        y_range=[0, 100],
    )
    for x in vertical_lines[:-1]:
        plotter2.draw_vline(x)
    
    plot.save_canvas(pads.c, f'{output_dir}/main_channel_yields')



##########################################################################################
###                                        MAIN                                        ###
##########################################################################################

def parse_args():
    parser = ArgumentParser(
        description="Plots various kinematic comparisons between the MC backgrounds and data for the VVSemileptonic analysis.", 
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('filepaths', nargs='+')
    parser.add_argument('-o', '--output', default='./output/plots')
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
    if not os.path.exists(args.output):
            os.makedirs(args.output)

    for k,v in RUN_FUNCS.items():
        v(file_manager, args.output)


if __name__ == "__main__":
    main()