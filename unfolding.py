#!/usr/bin/env python3
'''
@file unfolding.py
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch
@date April 27, 2023
@brief Unfolding postscripts for creating response matrices and ResonanceFinder histograms

To setup,
    setupATLAS
    lsetup "root recommended"
    lsetup "python centos7-3.9"
Note that this can't be setup at the same time with AnalysisBase or else you get a lot of
conflicts :(

To run
    unfolding.py path/to/reader/output.root SMVV 1
Also check the hardcoded variables and binning in [main]. This creates the following plots:
    migration_matrix.png/pdf 
    eff_acc.png/pdf
    fid_reco.png/pdf
and the following output file for RF input:
    rf_histograms.root
'''

from plotting import plot
import ROOT
import numpy as np
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import utils

##############################################################################
###                               UTILITIES                                ###
##############################################################################

def get_migration_matrix(h):
    '''
    Creates a migration matrix from an input 2d distribution of fid vs reco.

    @param h
        Input 2d histogram of fid vs reco. The y axis should be the reco distribution of
        a variable, and the x axis should be the fiducial distrubtion of the same variable.
    @return
        A new TH2F with matching bins as [h]. The underflows should not be used anymore.
    '''
    h = h.Clone()

    ### Remove negative bins ###
    tot_preclean = h.ProjectionX('_px', 1, h.GetNbinsY())
    for x in range(1, h.GetNbinsX() + 1):
        for y in range(1, h.GetNbinsY() + 1):
            v = h.GetBinContent(x, y)
            if v < 0:
                plot.warning("get_migration_matrix() Negative bin @ "
                             f"fid [{h.GetXaxis().GetBinLowEdge(x)}, {h.GetXaxis().GetBinLowEdge(x+1)}], " 
                             f"reco [{h.GetYaxis().GetBinLowEdge(y)}, {h.GetYaxis().GetBinLowEdge(y+1)}]: "
                             f"{v} / {tot_preclean.GetBinContent(x)} ({100 * v / tot_preclean.GetBinContent(x):.2f}%)")
                h.SetBinContent(x, y, 0)
    
    ### Normalize each column ###
    tot_fid = h.ProjectionX('_px', 1, h.GetNbinsY()) # The denominator should only be the "visible" bins
    for x in range(1, h.GetNbinsX() + 1):
        tot = tot_fid.GetBinContent(x)
        if tot <= 0: continue
        for y in range(1, h.GetNbinsY() + 1):
            v = h.GetBinContent(x, y)
            e = h.GetBinError(x, y)
            h.SetBinContent(x, y, v / tot)
            h.SetBinError(x, y, e / tot) # TODO is this right
    
    return h


def get_unfolding_efficiency(h):
    '''
    Creates an efficiency histogram from an input 2d distribution of fid vs reco.

    @param h
        Input 2d histogram of reco vs fid. The y axis should be the reco distribution of
        a variable, and the x axis should be the fiducial distrubtion of the same variable.
        The underflow of the y/x axes should be the mis-efficieny/acceptance respectively, the
        events that fall into only one of either reco/fiducial selections.

    @return
        A new TH1F with the bins matching the x axis bins of [h], containing the efficiency values.
    '''
    total = h.ProjectionX()
    eff = h.ProjectionX(h.GetName() + '_eff', 1, h.GetNbinsY()) # skip underflow bin == miseff
    eff.Divide(total)
    return eff


def get_unfolding_accuracy(h):
    '''
    Creates an acceptance histogram from an input 2d distribution of fid vs reco.

    @param h
        Input 2d histogram of reco vs fid. The y axis should be the reco distribution of
        a variable, and the x axis should be the fiducial distrubtion of the same variable.
        The underflow of the y/x axes should be the mis-efficieny/acceptance respectively, the
        events that fall into only one of either reco/fiducial selections.

    @return
        A new TH1F with the bins matching the y axis bins of [h], containing the acceptance values.
    '''
    total = h.ProjectionY()
    acc = h.ProjectionY(h.GetName() + '_acc', 1, h.GetNbinsX()) # skip underflow bin == misacc
    acc.Divide(total)
    return acc


def get_response_matrix(h):
    '''
    Creates the response matrix from an input 2d distribution of fid vs reco.

    @see [get_migration_matrix]
    '''
    mig_mtx = get_migration_matrix(h)
    efficiency = get_unfolding_efficiency(h)
    acceptance = get_unfolding_accuracy(h)

    for y in range(1, mig_mtx.GetNbinsY() + 1):
        for x in range(1, mig_mtx.GetNbinsX() + 1):
            mig = mig_mtx.GetBinContent(x, y)
            err = mig_mtx.GetBinError(x, y)
            eff = efficiency.GetBinContent(x)
            acc = acceptance.GetBinContent(y)
            if acc > 0:
                mig_mtx.SetBinContent(x, y, eff / acc * mig)
                mig_mtx.SetBinError(x, y, eff / acc * err) # TODO is this right?
            else:
                mig_mtx.SetBinContent(x, y, 0)
                mig_mtx.SetBinError(x, y, 0)

    return mig_mtx


def optimize_binning(
        h, fid_range,
        threshold_diag=0.8,
        threshold_err=0.1,
        monotonic_bin_sizes=False,
):
    '''
    Attempts to optimize the binning for unfolding such that migration matrix is
    well-behaved. It checks for the following conditions:
        1. The diagonal entries have at least [thresold_diag] fraction of the events.
        2. The fractional MC stat error of the fiducial distribution < [threshold_err] in
           each bin.

    @param h
        Input 2d histogram of fid vs reco. The y axis should be the reco distribution of a
        variable, and the x axis should be the fiducial distrubtion of the same variable.
    @param fid_range
        The final range that should be covered by the bins.
    '''
    ### Range ###
    h_1d = h.ProjectionX()
    i_start = h_1d.FindFixBin(fid_range[0])
    i_end = h_1d.FindFixBin(fid_range[1])
    i = i_start

    ### Denominator in truth bin ###
    h_fid = h.ProjectionX('_px', i_start, i_end - 1) # end is inclusive :)

    ### Running bins ###
    bin_edges = [fid_range[0]]
    bin_indices = [i] # unused...
    last_merge_size = 0

    ### Loop once per NEW (merged) bin ###
    while i < i_end:
        ### Find size of this bin ###
        n_tot = 0
        n_diag = 0
        err_tot = 0
        merge_size = 0
        while (n_diag <= 0
               or n_diag / n_tot < threshold_diag
               or (monotonic_bin_sizes and merge_size < last_merge_size)
               or err_tot**0.5 / n_tot > threshold_err):
            
            ### Last bin incomplete ###
            if i + merge_size >= i_end:
                # Merge it with the previously completed bin by popping the last edge
                bin_edges.pop()
                bin_indices.pop()
                break

            ### Merge diagonal ###
            # This is essentially adding "L" blocks 
            n_tot += h_fid.GetBinContent(i + merge_size)
            err_tot += h_fid.GetBinError(i + merge_size)**2
            # First add top of "L"
            for x in range(i, i + merge_size):
                n_diag += h.GetBinContent(x, i + merge_size)
            # Then add right of "L"
            for y in range(i, i + merge_size + 1):
                n_diag += h.GetBinContent(i + merge_size, y)
            merge_size += 1

        ### Add end of this bin ###
        i += merge_size
        bin_edges.append(h.GetXaxis().GetBinLowEdge(i))
        bin_indices.append(i)
        last_merge_size = merge_size

    print('Optimized bins:', h.GetName(), [int(x) for x in bin_edges])
    return bin_edges



##############################################################################
###                                PLOTTING                                ###
##############################################################################

def plot_migration_matrix(mtx, var, **kwargs):
    '''
    Plots the 2D migration matrix in COLZ mode with overlaid text.
    '''
    ROOT.gStyle.SetPaintTextFormat('2.0f')
    ROOT.gStyle.SetPalette(ROOT.kAlpine)

    ### For plotting, hide small bins. Otherwise colz shows a sea of blue ###
    h = get_migration_matrix(mtx)
    for y in range(1, h.GetNbinsY() + 1):
        for x in range(1, h.GetNbinsX() + 1):
            v = h.GetBinContent(x, y)
            if v < 0.01:
                h.SetBinContent(x, y, 0)
            else:
                h.SetBinContent(x, y, v * 100)

    ### Draw ###
    plot.plot([h],
        ytitle=f'Detector {var:title}',
        xtitle=f'Fiducial {var:title}',
        opts='COLZ TEXT',
        **kwargs,
    )


def plot_eff_acc(mtx, var, **kwargs):
    efficiency = get_unfolding_efficiency(mtx)
    accuracy = get_unfolding_accuracy(mtx)
    plot.plot_2panel([efficiency], [accuracy],
        xtitle=f'{var:title}',
        ytitle='Efficiency',
        ytitle2='#splitline{Fiducial}{Accuracy}',
        # y_range=[0, 0.5],
        # y_range2=[0.5, 1],
        **kwargs
    )


def plot_fid_reco(mtx, var, **kwargs):
    fid = mtx.ProjectionX()
    reco = mtx.ProjectionY()
    int_reco = mtx.ProjectionY('int_reco', 1, mtx.GetNbinsY())

    fid.Scale(1, 'width')
    reco.Scale(1, 'width')
    int_reco.Scale(1, 'width')

    ratio = fid.Clone()
    ratio.Divide(reco)

    plot.plot_ratio([fid, reco, int_reco], [ratio],
        legend=['Fiducial', 'Detector', 'Both'],
        ytitle='Events / Bin Width',
        ytitle2='#frac{Fiducial}{Detector}',
        xtitle=f'{var:title}',
        linecolor=plot.colors.tableu,
        markercolor=plot.colors.tableu,
        logy=True,
        **kwargs,
    )




##############################################################################
###                                  MAIN                                  ###
##############################################################################

def get_bins(sample, lepton_channel, var: utils.Variable):
    if var.name == "vv_m":
        if lepton_channel == '0':
            # optimized binning with threshold_diag=0.8, threshold_err=0.1, monotonic_bin_sizes=False
            return [500, 740, 930, 1160, 1440, 1800, 2230, 3000]
        elif lepton_channel == '1':
            return [500, 600, 700, 800, 900, 1020, 1170, 1310, 1470, 1780, 2090, 2400, 3000]
        elif lepton_channel == '2':
            # optimized binning with threshold_diag=0.8, threshold_err=0.1, monotonic_bin_sizes=False
            return [500, 580, 680, 780, 900, 1050, 1220, 1410, 1680, 1910, 2210, 3000]
    raise NotImplementedError()


def main():
    ### Args ###
    parser = ArgumentParser(
        description="Plots the migration matrix, efficiency and fiducial accuracy. Saves the response matrix as histograms for use in ResonanceFinder.", 
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('filepath', help='Path to the CxAODReader output histograms')
    parser.add_argument('sample', help='Sample name used in CxAODReader, such as "SMVV"')
    parser.add_argument("lepton", choices=['0', '1', '2'])
    parser.add_argument('-o', '--output', default='./output')
    parser.add_argument('--optimizeToRange', help='Automatically optimize the binning. Pass in a "min,max" range of values that the binning should cover.')
    args = parser.parse_args()

    ### Output dir ###
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    ### Config ###
    vars = [utils.vv_m]
    common_subtitle = [
        '#sqrt{s}=13 TeV, 140 fb^{-1}',
        f'{args.lepton}-lepton channel, {args.sample}',
    ]
    output_basename = f'{args.output}/{args.sample}_{args.lepton}lep'
    plot.file_formats = ['png', 'pdf']

    ### Files ###    
    f = ROOT.TFile(args.filepath)
    rf_output_path = f'{output_basename}_rf_histograms.root'
    rf_output_file = ROOT.TFile(rf_output_path, 'RECREATE')

    ### Run ###
    for var in vars:
        ### Get base histogram ###
        mtx = f.Get(f'{args.sample}_VV{args.lepton}Lep_Merg_unfoldingMtx_{var}_recoWeight')

        ### Rebin ###
        if args.optimizeToRange:
            bins = optimize_binning(mtx, [float(x) for x in args.optimizeToRange.split(',')])
        else:
            bins = get_bins(args.sample, args.lepton, var)
        mtx = plot.rebin2d(mtx, bins, bins)
    
        ### Plot ###
        plot_migration_matrix(mtx, var,
            filename=f'{output_basename}_{var}_migration_matrix',
            subtitle=[
                *common_subtitle,
                '% migration from each fiducial bin'
            ],
        )
        plot_eff_acc(mtx, var,
            filename=f'{output_basename}_{var}_eff_acc',
            subtitle=common_subtitle,
        )
        plot_fid_reco(mtx, var,
            filename=f'{output_basename}_{var}_fid_reco',
            subtitle=common_subtitle,
        )

        ### Save ###
        reponse_matrix = get_response_matrix(mtx)
        for x in range(1, reponse_matrix.GetNbinsX() + 1):
            p = reponse_matrix.ProjectionY(reponse_matrix.GetName() + f'_projY_{x}', x, x)
            p.Write(f'ResponseMatrix_{var}_fid' + str(x).rjust(2, '0'))
    
    plot.success(f'Saved response matrix histograms to {rf_output_path}')



if __name__ == "__main__":
    main()