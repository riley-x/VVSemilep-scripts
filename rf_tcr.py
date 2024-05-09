#!/usr/bin/env python3
'''
@file rf_tcr.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date Oct 19, 2023 
@brief Script for running ResonanceFinder fits to the TCR region.

TODO out of date
'''

import ROOT
from ROOT import RF as RF
import numpy as np
import sys


class Sample:
    def __init__(self, name, file_path, hist_name, *scale_args):
        self.name = name
        self.file_path = file_path
        self.hist_name = hist_name
        self.scale_args = list(scale_args) # name, start_val, min_val, max_val, rf_mode

    def add_scale(self, *args):
        '''
        args: name, start_val, min_val, max_val, rf_mode
        '''
        self.scale_args.append(args)

    def add_to(self, channel):
        channel.addSample(self.name, self.file_path, self.hist_name.format(region=channel.name()))
        self.rf_obj = channel.sample(self.name)
        for args in self.scale_args:
            self.rf_obj.multiplyBy(*args)
        return self.rf_obj


def main():
    ### Config ###
    root_dir = '/home/rileyx/VVSemilep/user/1lep'
    reader_version = 'Sep23-ANN'
    var = 'vv_m'
    bins = np.array([500, 700, 810, 940, 1090, 1260, 1500, 1750, 2000, 2500, 6000], dtype=float)

    ### Start analysis and define directory structure ###
    RFAnalysis = RF.DefaultAnalysisRunner(f'1lep_TCR-only_{var}')
    RFAnalysis.setOutputDir('output')
    RFAnalysis.setOutputWSTag(reader_version)
    # RFAnalysis.setCollectionTagNominal("Nominal")
    # RFAnalysis.setCollectionTagUp("_up")
    # RFAnalysis.setCollectionTagDown("_dn")
    # RFAnalysis.setExpert(True)

    ### Lumi uncertainties ###
    lumi_scale=1.
    lumi_uncert=0.017
    lumi_args = ['lumiNP', lumi_scale, lumi_scale*(1-lumi_uncert), lumi_scale*(1+lumi_uncert), RF.MultiplicativeFactor.GAUSSIAN]

    ### Regions ###
    regions = ['kin_ptJ270_hp_tcr']

    ### Samples ###
    mc_samples = {
        # 'Zjets':   Sample('Zjets',   f'{root_dir}/1lep_Zjets_x_{reader_version}.reco.root',   '*_{region}_vv_m_rebin'),
        'Wjets':   Sample('Wjets',   f'{root_dir}/1lep_Wjets_Sherpa2211_x_{reader_version}.reco.root',   '*__R_{region}_' + var + '$'),
        'diboson': Sample('diboson', f'{root_dir}/1lep_diboson_Sherpa2211_x_{reader_version}.reco.root', '*__R_{region}_' + var + '$'),
        'ttbar':   Sample('ttbar',   f'{root_dir}/1lep_ttbar_x_{reader_version}.reco.root',              '*__R_{region}_' + var + '$'),
        'stop':    Sample('stop',    f'{root_dir}/1lep_stop_x_{reader_version}.reco.root',               '*__R_{region}_' + var + '$'),
    }
    # backgrounds['Zjets'].add_scale("XS_Zjets", 1.0, 0.5, 1.5, RF.MultiplicativeFactor.FREE)
    # mc_samples['Wjets'].add_scale("XS_Wjets", 1.0, 0.5, 1.5, RF.MultiplicativeFactor.FREE)
    mc_samples['ttbar'].add_scale("XS_ttbar", 1.0, 0.5, 1.5, RF.MultiplicativeFactor.FREE)
    mc_samples['stop'].add_scale("XS_stop",   1.0, 0.8, 1.2, RF.MultiplicativeFactor.GAUSSIAN)
    RFAnalysis.defineSignal('ttbar')
    # RFAnalysis.defineSignal('stop')
    RFAnalysis.setPOI("XS_ttbar")
    # RFAnalysis.setPOI("XS_stop")
    
    ### Loop over regions ###
    for region in regions:
        RFAnalysis.rebinRegion(region, len(bins)-1, bins)
        RFAnalysis.addChannel(region)
        channel = RFAnalysis.channel(region)
        channel.setStatErrorThreshold(0.01) # Add MC stat uncertainty

        ### Data ###
        if False:
            RFAnalysis.fakeData(region, 1) # includeSignal
        else:
            channel.addData('data', f'{root_dir}/1lep_data_x_{reader_version}.reco.root', f'data__R_{region}_{var}')

        ### Add lumi,stat error to all MC samples ###
        for _,sample in mc_samples.items():
            srf = sample.add_to(channel)
            srf.multiplyBy(*lumi_args)
            srf.setUseStatError(True) # Consider the MC uncertainity on this sample

    ### Run ###
    RFAnalysis.produceWS()



if __name__ == "__main__":
    main()
