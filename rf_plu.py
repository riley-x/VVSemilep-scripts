#!/usr/bin/env python
import ROOT
from ROOT import RF

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import utils


def run(
        file_manager : utils.FileManager,
        lepton_channel : int,
        variable : utils.Variable,
        bins : list[float],
        response_matrix_path : str,
        output_dir : str,
        mu_stop : tuple[float, float],
        mu_ttbar : tuple[float, float],
    ):
    ### Config ###
    analysis = 'VVUnfold'
    outputWSTag = 'test'
    lumi_uncert = 0.017
  
    ### Define the workspace ###
    VVUnfold = RF.DefaultAnalysisRunner(analysis)
    VVUnfold.setOutputDir(output_dir + '/rf')
    VVUnfold.setOutputWSTag(outputWSTag)

    ### Single region ###
    region = "SR"
    VVUnfold.addChannel(region)
    VVUnfold.channel(region).setStatErrorThreshold(0.05) # 0.05 means that errors < 5% will be ignored

    ### Get hists ###
    hist_name = '*_VV1Lep_MergHP_Inclusive_SR_' + utils.generic_var_to_lep(variable, lepton_channel).name
    # WARNING this assumes each sample is in separate files!

    ### Add data ###
    file_names = file_manager.get_file_names(lepton_channel, utils.Sample.data)
    VVUnfold.channel(region).addData('data', file_names)
    VVUnfold.channel(region).data().setSearchStrings(hist_name)

    ### Add ttbar ###
    file_names = file_manager.get_file_names(lepton_channel, utils.Sample.ttbar)
    VVUnfold.channel(region).addSample('ttbar', file_names)
    sample = VVUnfold.channel(region).sample('ttbar')
    sample.setSearchStrings(hist_name)
    sample.multiplyBy('mu-ttbar', mu_ttbar[0], mu_ttbar[0] - mu_ttbar[1], mu_ttbar[0] + mu_ttbar[1], RF.MultiplicativeFactor.GAUSSIAN)
    sample.multiplyBy('lumiNP',  1, 1 - lumi_uncert, 1 + lumi_uncert, RF.MultiplicativeFactor.GAUSSIAN)
    sample.setUseStatError(True)

    ### Add stop ###
    file_names = file_manager.get_file_names(lepton_channel, utils.Sample.stop)
    VVUnfold.channel(region).addSample('stop', file_names)
    sample = VVUnfold.channel(region).sample('stop')
    sample.setSearchStrings(hist_name)
    sample.multiplyBy('mu-stop', mu_stop[0], mu_stop[0] - mu_stop[1], mu_stop[0] + mu_stop[1], RF.MultiplicativeFactor.GAUSSIAN)
    sample.multiplyBy('lumiNP',  1, 1 - lumi_uncert, 1 + lumi_uncert, RF.MultiplicativeFactor.GAUSSIAN)
    sample.setUseStatError(True)

    ### Add GPR ###
    VVUnfold.channel(region).addSample('vjets', output_dir + '/gpr/gpr_' + str(lepton_channel) + 'lep_vjets_yield.root')
    sample = VVUnfold.channel(region).sample('vjets')
    sample.setSearchStrings('Vjets_SR_' + variable.name)
    sample.setUseStatError(True)

    ### Add signals ###
    for i in range(1, len(bins) + 1):
        i_str = str(i).rjust(2, '0')
        sigName = "bin" + i_str
        poiName = "mu_" + i_str

        VVUnfold.channel(region).addSample(sigName, response_matrix_path, "ResponseMatrix_fid" + i_str)
        VVUnfold.channel(region).sample(sigName).multiplyBy(poiName, 0, 0, 1e6)
        VVUnfold.defineSignal(VVUnfold.channel(region).sample(sigName), "Unfold")
        VVUnfold.addPOI(poiName)

    ### Make workspace ###
    #VVUnfold.reuseHist(True) # What is this for?
    VVUnfold.linearizeRegion(region) # What is this for?
    VVUnfold.debugPlots(True)
    VVUnfold.produceWS()

##########################################################################################
###                                        MAIN                                        ###
##########################################################################################

def parse_args():
    parser = ArgumentParser(
        description="", 
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('filepaths', nargs='+')
    parser.add_argument("--lepton", required=True, type=int, choices=[0, 1, 2])
    parser.add_argument("--var", required=True, help='Variable to fit against; this must be present in the utils.py module.')
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
    '''
    See file header.
    '''
    args = parse_args()
    file_manager = get_files(args.filepaths)
    var = getattr(utils.Variable, args.var)
    run(
        file_manager=file_manager,
        lepton_channel=args.lepton,
        variable=var,
        bins=utils.get_bins(args.lepton, var),
        response_matrix_path=args.output + '/response_matrix/diboson_' + str(args.lepton) + 'lep_rf_histograms.root',
        output_dir=args.output,
        mu_stop=(1, 0.2),
        mu_ttbar=(0.72, 0.03),
    )

if __name__ == '__main__':
    main()