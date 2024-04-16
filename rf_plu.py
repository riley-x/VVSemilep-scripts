#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import shutil

import utils
from plotting import plot

##########################################################################################
###                                       CONFIG                                       ###
##########################################################################################

def ws_path(output_dir, lep, var, mode, stat_validation_index=None):
    if stat_validation_index is None:
        return f'{output_dir}/rf/{lep}lep_{var}_{mode}.ws.root'
    else:
        return f'{output_dir}/rf/{lep}lep_{var}_{mode}.ws_var{stat_validation_index:03}.root'

##########################################################################################
###                                        MODES                                       ###
##########################################################################################

def _add_plu(runner, lepton_channel, variable, region, response_matrix_path, **_):
    bins = utils.get_bins(lepton_channel, variable)
    nbins = len(bins) - 1
    signal_names = ''
    for i in range(1, nbins + 1):
        i_str = str(i).rjust(2, '0')
        sigName = "bin" + i_str
        poiName = "mu_" + i_str
        if i > 1:
            signal_names += '-'
        signal_names += sigName

        runner.channel(region).addSample(sigName, response_matrix_path, f'ResponseMatrix_{variable}_fid{i_str}')
        runner.channel(region).sample(sigName).multiplyBy(poiName, 100, 0, 1e6)
        runner.defineSignal(runner.channel(region).sample(sigName), 'Unfold')
        runner.addPOI(poiName)
    return signal_names

def _add_diboson(runner, lumi_uncert, region, hist_file_format, hist_name, **_):
    from ROOT import RF # type: ignore

    runner.channel(region).addSample('diboson', hist_file_format.format(sample='diboson'), hist_name.format('diboson'))
    sample = runner.channel(region).sample('diboson')
    sample.multiplyBy(utils.variation_lumi, 1, 1 - lumi_uncert, 1 + lumi_uncert, RF.MultiplicativeFactor.GAUSSIAN)
    sample.multiplyBy('mu-diboson', 1, 0, 5, RF.MultiplicativeFactor.FREE)
    sample.setUseStatError(True)
    for variation in utils.variations_hist:
        sample.addVariation(variation)

    runner.defineSignal(sample, 'diboson')
    runner.addPOI('mu-diboson')
    return 'diboson'


##########################################################################################
###                                     COMMON RUN                                     ###
##########################################################################################
    
def run(
        mode : str,
        lepton_channel : int,
        variable : utils.Variable,
        response_matrix_path : str,
        output_dir : str,
        hist_file_format : str,
        mu_stop : tuple[float, float],
        mu_ttbar : tuple[float, float],
        gpr_mu_corrs : bool = True,
        stat_validation_index : int = None,
    ):
    '''
    @param mode
        What should be set as the signal.
            - 'PLU': Runs the profile-likelihood unfolding fit, such that there are nbins
              signals, and the strength of each signal is the fiducial event count. These
              use the response matrix histograms as the signal inputs.
            - 'diboson': Runs a direct fit to the diboson signal strength, with the
              diboson MC as the only floating signal. Note this only looks at a single
              channel at a time though.
            - an EFT term: TODO
    '''
    from ROOT import RF # type: ignore

    ### Config ###
    analysis = f'VV{lepton_channel}lep'
    outputWSTag = 'feb24'
    lumi_uncert = 0.017
    if stat_validation_index is not None:
        outputWSTag += f'_var{stat_validation_index:03}'
  
    ### Define the workspace ###
    runner = RF.DefaultAnalysisRunner(analysis)
    runner.setOutputDir(output_dir + '/rf')
    runner.setOutputWSTag(outputWSTag)
    runner.setCollectionTagNominal('')
    runner.setCollectionTagUp(utils.variation_up_key)
    runner.setCollectionTagDown(utils.variation_down_key)
    # VVUnfold.setPruningThreshold(0.01, 0.01) # is this buggy? According to Liza on Mattermost

    ### Single region ###
    region = "SR"
    runner.addChannel(region)
    runner.channel(region).setStatErrorThreshold(0.05) # 0.05 means that errors < 5% will be ignored

    ### Get hists ###
    hist_name = f'{{}}_VV{lepton_channel}Lep_MergHP_Inclusive_SR_' + utils.generic_var_to_lep(variable, lepton_channel).name
    # WARNING this assumes each sample is in separate files!

    ### Add data ###
    if stat_validation_index == None:
        hist_name_data = hist_name.format('data')
    else:
        hist_name_data = hist_name.format(f'data_var{stat_validation_index:03}')
    runner.channel(region).addData('data', hist_file_format.format(sample='data'), hist_name_data)

    ### Add ttbar ###
    runner.channel(region).addSample('ttbar', hist_file_format.format(sample='ttbar'), hist_name.format('ttbar'))
    sample = runner.channel(region).sample('ttbar')
    # ResonanceFinder has a major bug when using mean != 1 GAUSSIAN constraints. So hack
    # fix by scaling by the mean as a constant first (which works fine).
    sample.multiplyBy('mu-ttbar_nom', mu_ttbar[0])
    sample.multiplyBy(utils.variation_mu_ttbar, 1, 1 - mu_ttbar[1] / mu_ttbar[0], 1 + mu_ttbar[1] / mu_ttbar[0], RF.MultiplicativeFactor.GAUSSIAN)
    sample.multiplyBy(utils.variation_lumi, 1, 1 - lumi_uncert, 1 + lumi_uncert, RF.MultiplicativeFactor.GAUSSIAN)
    sample.setUseStatError(True)
    for variation in utils.variations_hist:
        sample.addVariation(variation)

    ### Add stop ###
    runner.channel(region).addSample('stop', hist_file_format.format(sample='stop'), hist_name.format('stop'))
    sample = runner.channel(region).sample('stop')
    # ResonanceFinder has a major bug when using mean != 1 GAUSSIAN constraints. So hack
    # fix by scaling by the mean as a constant first (which works fine).
    sample.multiplyBy('mu-stop_nom', mu_stop[0])
    sample.multiplyBy(utils.variation_mu_stop, 1, 1 - mu_stop[1] / mu_stop[0], 1 + mu_stop[1] / mu_stop[0], RF.MultiplicativeFactor.GAUSSIAN)
    sample.multiplyBy(utils.variation_lumi,  1, 1 - lumi_uncert, 1 + lumi_uncert, RF.MultiplicativeFactor.GAUSSIAN)
    sample.setUseStatError(True)
    for variation in utils.variations_hist:
        sample.addVariation(variation)

    ### Add GPR ###
    runner.channel(region).addSample('vjets', f'{output_dir}/gpr/gpr_{lepton_channel}lep_vjets_yield.root', 'Vjets_SR_' + variable.name)
    sample = runner.channel(region).sample('vjets')
    sample.setUseStatError(True)
    if gpr_mu_corrs:
        for variation in utils.variations_custom:
            sample.addVariation(variation)
        for variation in utils.variations_hist:
            sample.addVariation(variation)

    ### Mode switch (signals and diboson background) ###
    common_args = {
        'runner': runner,
        'lepton_channel': lepton_channel,
        'variable': variable,
        'region': region,
        'hist_file_format': hist_file_format,
        'hist_name': hist_name,
        'lumi_uncert': lumi_uncert,
    }
    if mode == 'PLU':
        signal_name = _add_plu(**common_args, response_matrix_path=response_matrix_path)
    elif mode == 'diboson':
        signal_name = _add_diboson(**common_args)
    else:
        raise RuntimeError(f'rf.py() unknown mode {mode}')

    ### Global options ###
    #VVUnfold.reuseHist(True) # What is this for?
    runner.linearizeRegion(region) # What is this for?
    runner.debugPlots(True)

    ### Make workspace ###
    if stat_validation_index is None:
        log_name = f'{output_dir}/rf/log.{lepton_channel}lep_{variable}_{mode}.rf.txt'
    else:
        log_name = f'{output_dir}/rf/log.{lepton_channel}lep_{variable}_{mode}.rf_var{stat_validation_index:03}.txt'
    with plot.redirect(log_name):
        runner.produceWS()

    ### Copy back ###
    rf_output_path = f'{output_dir}/rf/ws/{analysis}_{signal_name}_{outputWSTag}.root'
    target_path = ws_path(output_dir, lepton_channel, variable, mode, stat_validation_index)
    shutil.copyfile(rf_output_path, target_path)
    plot.success(f'Created workspace at {target_path}')
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
    parser.add_argument("--mode", required=True, choices=['PLU', 'diboson'])
    return parser.parse_args()


def main():
    '''
    See file header.
    '''
    args = parse_args()
    var = getattr(utils.Variable, args.var)
    run(
        mode=args.mode,
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