#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import shutil

import utils
from plotting import plot

def ws_path(output_dir, lep, var):
    return f'{output_dir}/rf/{lep}lep_{var}.plu_ws.root'

def run(
        lepton_channel : int,
        variable : utils.Variable,
        response_matrix_path : str,
        output_dir : str,
        hist_file_format : str,
        mu_stop : tuple[float, float],
        mu_ttbar : tuple[float, float],
    ):
    from ROOT import RF

    ### Config ###
    analysis = 'VVUnfold'
    outputWSTag = 'feb24'
    lumi_uncert = 0.017
  
    ### Define the workspace ###
    VVUnfold = RF.DefaultAnalysisRunner(analysis)
    VVUnfold.setOutputDir(output_dir + '/rf')
    VVUnfold.setOutputWSTag(outputWSTag)
    # VVUnfold.setCollectionTagNominal("")
    # VVUnfold.setCollectionTagUp("_up")
    # VVUnfold.setCollectionTagDown("_down")

    ### Single region ###
    region = "SR"
    VVUnfold.addChannel(region)
    VVUnfold.channel(region).setStatErrorThreshold(0.05) # 0.05 means that errors < 5% will be ignored

    ### Get hists ###
    hist_name = f'*_VV{lepton_channel}Lep_MergHP_Inclusive_SR_' + utils.generic_var_to_lep(variable, lepton_channel).name
    # WARNING this assumes each sample is in separate files!

    ### Add data ###
    VVUnfold.channel(region).addData('data', hist_file_format.format(sample='data'), hist_name)

    ### Add ttbar ###
    VVUnfold.channel(region).addSample('ttbar', hist_file_format.format(sample='ttbar'), hist_name)
    sample = VVUnfold.channel(region).sample('ttbar')
    # ResonanceFinder has a major bug when using mean != 1 GAUSSIAN constraints. So hack
    # fix by scaling by the mean as a constant first (which works fine).
    sample.multiplyBy('mu-ttbar_nom', mu_ttbar[0])
    sample.multiplyBy('mu-ttbar', 1, 1 - mu_ttbar[1] / mu_ttbar[0], 1 + mu_ttbar[1] / mu_ttbar[0], RF.MultiplicativeFactor.GAUSSIAN)
    sample.multiplyBy('lumiNP', 1, 1 - lumi_uncert, 1 + lumi_uncert, RF.MultiplicativeFactor.GAUSSIAN)
    sample.setUseStatError(True)
    for variation in utils.variations_hist:
        sample.addVariation(variation)

    ### Add stop ###
    VVUnfold.channel(region).addSample('stop', hist_file_format.format(sample='stop'), hist_name)
    sample = VVUnfold.channel(region).sample('stop')
    # ResonanceFinder has a major bug when using mean != 1 GAUSSIAN constraints. So hack
    # fix by scaling by the mean as a constant first (which works fine).
    sample.multiplyBy('mu-stop_nom', mu_stop[0])
    sample.multiplyBy('mu-stop', 1, 1 - mu_stop[1] / mu_stop[0], 1 + mu_stop[1] / mu_stop[0], RF.MultiplicativeFactor.GAUSSIAN)
    sample.multiplyBy('lumiNP',  1, 1 - lumi_uncert, 1 + lumi_uncert, RF.MultiplicativeFactor.GAUSSIAN)
    sample.setUseStatError(True)
    for variation in utils.variations_hist:
        sample.addVariation(variation)

    ### Add GPR ###
    VVUnfold.channel(region).addSample('vjets', f'{output_dir}/gpr/gpr_{lepton_channel}lep_vjets_yield.root', 'Vjets_SR_' + variable.name)
    sample = VVUnfold.channel(region).sample('vjets')
    sample.setUseStatError(True)
    for variation in utils.variations_custom:
        sample.addVariation(variation)
    for variation in utils.variations_hist:
        sample.addVariation(variation)

    ### Add signals ###
    bins = utils.get_bins(lepton_channel, variable)
    nbins = len(bins) - 1
    for i in range(1, nbins + 1):
        i_str = str(i).rjust(2, '0')
        sigName = "bin" + i_str
        poiName = "mu_" + i_str

        VVUnfold.channel(region).addSample(sigName, response_matrix_path, f'ResponseMatrix_{variable}_fid{i_str}')
        VVUnfold.channel(region).sample(sigName).multiplyBy(poiName, 100, 0, 1e6)
        VVUnfold.defineSignal(VVUnfold.channel(region).sample(sigName), 'Unfold')
        VVUnfold.addPOI(poiName)

    ### Make workspace ###
    #VVUnfold.reuseHist(True) # What is this for?
    VVUnfold.linearizeRegion(region) # What is this for?
    VVUnfold.debugPlots(True)
    with plot.redirect(f'{output_dir}/rf/log.{lepton_channel}lep_{variable}.rf.txt'):
        VVUnfold.produceWS()

    ### Copy back ###
    rf_output_path = f'{output_dir}/rf/ws/VVUnfold_'
    for i in range(1, nbins + 1):
        if i > 1:
            rf_output_path += '-'
        rf_output_path += 'bin' + str(i).rjust(2, '0')
    rf_output_path += f'_{outputWSTag}.root'

    target_path = ws_path(output_dir, lepton_channel, variable)
    shutil.copyfile(rf_output_path, target_path)
    plot.success(f'Created PLU workspace at {target_path}')
    return target_path

##########################################################################################
###                                        MAIN                                        ###
##########################################################################################

def parse_args():
    parser = ArgumentParser(
        description="", 
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--lepton", required=True, type=int, choices=[0, 1, 2])
    parser.add_argument("--var", required=True, help='Variable to fit against; this must be present in the utils.py module.')
    parser.add_argument('-o', '--output', default='./output')
    return parser.parse_args()


def main():
    '''
    See file header.
    '''
    args = parse_args()
    var = getattr(utils.Variable, args.var)
    run(
        lepton_channel=args.lepton,
        variable=var,
        response_matrix_path=f'{args.output}/response_matrix/diboson_{args.lepton}lep_rf_histograms.root',
        output_dir=args.output,
        hist_file_format=f'{args.output}/rebin/{args.lepton}lep_{{sample}}_rebin.root',
        mu_stop=(1, 0.2),
        mu_ttbar=(0.72, 0.03),
    )

if __name__ == '__main__':
    main()