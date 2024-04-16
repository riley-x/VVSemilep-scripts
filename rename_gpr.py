#!/usr/bin/env python3
'''
@file rename_gpr.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date April 16, 2024 
@brief Renames GPR histograms in ROOT files to match the resonance finder side. Also adds
the nominal MC V+jets TCR to the file.

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------

Pass the path to an master.py output directory (parent directory of the gpr and rebin
folders), and optionally a naming scheme for the output file.

    rename_gpr.py ./output gpr_{lep}lep_vjets.root

'''

import ROOT # type: ignore
import sys


##########################################################################################
###                                        MAIN                                        ###
##########################################################################################


def main():
    root_dir = sys.argv[1]
    if len(sys.argv) > 2:
        out_name = sys.argv[2]
    else:
        out_name = 'gpr_{lep}lep_vjets.rename.root'

    for lep in [0, 1, 2]:
        f_gpr = ROOT.TFile(f'{root_dir}/gpr/gpr{lep}lep_vjets_yield.root')
        f_wjets = ROOT.TFile(f'{root_dir}/rebin/{lep}lep_wjets_rebin.root')
        f_zjets = ROOT.TFile(f'{root_dir}/rebin/{lep}lep_zjets_rebin.root')
        f_out = ROOT.TFile(out_name.format(lep=lep))
    
        ### Get GPR histo names ###
        used_names = set() # ignore backup cycles
        for key in f_gpr.GetListOfKeys():
            name = key.GetName()
            if '__mu-' in name: continue # skip mu correlations for now
            used_names.add(name)

        ### Copy and rename GPR histos ###
        for name in sorted(used_names):
            h = f_gpr.Get(name)
            if lep == 0:
                name = name.replace('Vjets_SR', 'Vjets_VV0Lep_MergHP_Inclusive_SR')
                name = name.replace('vv_mt', 'vvJ_mT')
            elif lep == 1:
                name = name.replace('Vjets_SR', 'Vjets_VV1Lep_MergHP_Inclusive_SR')
                name = name.replace('vv_m', 'lvJ_m')
            else:
                name = name.replace('Vjets_SR', 'Vjets_VV2Lep_MergHP_Inclusive_SR')
                name = name.replace('vv_m', 'llJ_m')
            # name = name.replace('__mu-', '_SysMu-')
            name = name.replace('__Sys', '_Sys')
            
            f_out.cd()
            h.SetName(name)
            h.Write()

        ### Copy and rename TCR MC histos ###
        for key in f_wjets.GetListOfKeys():
            name = key.GetName()
            if 'Inclusive_SR' in name: continue

            h = f_wjets.Get(name).Clone()
            h.Add(f_zjets.Get(name.replace('wjets', 'zjets')))

            name = name.replace('wjets', 'Vjets')
            name = name.replace('__Sys', '_Sys')

            f_out.cd()
            h.SetName(name)
            h.Write()
        
        f_out.Write()
        f_out.Close()
        print(f"Saved {len(used_names)} histograms to {f_out.GetName()}")


if __name__ == "__main__":
    main()