#!/usr/bin/env python3
'''
@file merge_gpr_condor.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date April 16, 2024 
@brief Renames GPR histograms in ROOT files to match the resonance finder side

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------

Pass the path to a gpr file.

'''

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import ROOT # type: ignore
from pathlib import Path

##########################################################################################
###                                        MAIN                                        ###
##########################################################################################

def parse_args():
    parser = ArgumentParser(
        description="Merges the CSV files of condor GPR runs.", 
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('filepath')
    parser.add_argument('output')
    return parser.parse_args()

def main():
    args = parse_args()
    f_in = ROOT.TFile(args.filepath)
    f_out = ROOT.TFile(args.output or Path(args.filepath).stem + '.rename.root', 'RECREATE')
    
    used_names = set() # ignore backup cycles
    for key in f_in.GetListOfKeys():
        name = key.GetName()
        if '__mu-' in name: continue
        used_names.add(name)

    for name in sorted(used_names):
        h = f_in.Get(name)
        if '0lep' in args.filepath:
            name = name.replace('Vjets_SR', 'Wjets_VV0Lep_MergHP_Inclusive_SR')
            name = name.replace('vv_mt', 'vvJ_mT')
        elif '1lep' in args.filepath:
            name = name.replace('Vjets_SR', 'Wjets_VV1Lep_MergHP_Inclusive_SR')
            name = name.replace('vv_m', 'lvJ_m')
        else:
            name = name.replace('Vjets_SR', 'Wjets_VV2Lep_MergHP_Inclusive_SR')
            name = name.replace('vv_m', 'llJ_m')
        # name = name.replace('__mu-', '_SysMu-')
        name = name.replace('__Sys', '_Sys')
        
        f_out.cd()
        h.SetName(name)
        h.Write()
    
    f_out.Write()
    f_out.Close()
    print(f"Saved {len(used_names)} histograms to {f_out.GetName()}")


if __name__ == "__main__":
    main()