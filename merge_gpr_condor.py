#!/usr/bin/env python3
'''
@file merge_gpr_condor.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date April 2, 2024 
@brief Merges the CSV and ROOT files of condor GPR runs

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------

Pass the root of the output direction (i.e. parent of the 'gpr' directory created by
master.py)

    merge_gpr_condor.py ./output
'''

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import ROOT

##########################################################################################
###                                        MAIN                                        ###
##########################################################################################

def parse_args():
    parser = ArgumentParser(
        description="Merges the CSV files of condor GPR runs.", 
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('output_dir')
    return parser.parse_args()

def main():
    args = parse_args()
    base_dir = f'{args.output_dir}/gpr'
    csv_name = 'gpr_fit_results.csv'
    root_name = 'gpr_{lep}lep_vjets_yield.root'

    root_out_files = []
    for lep in [0, 1, 2]:
        root_out_files.append(ROOT.TFile(os.path.join(base_dir, root_name.format(lep=lep)), 'UPDATE'))
        
    first_line = True
    with open(os.path.join(base_dir, csv_name), 'w') as out_file:
        for dirpath, dirnames, filenames in os.walk(base_dir):
            if dirpath == base_dir: continue
            if csv_name in filenames:
                ### Write to CSV file ###
                with open(os.path.join(dirpath, csv_name), 'r') as in_file:
                    for i,line in enumerate(in_file):
                        if first_line:
                            out_file.write(line)
                            first_line = False
                        elif i > 0: # skip header
                            out_file.write(line)

                ### Write to ROOT file ###
                if '0lep' in dirpath:
                    lep = 0
                elif '1lep' in dirpath:
                    lep = 1
                else:
                    lep = 2
                
                f = ROOT.TFile(os.path.join(dirpath, root_name.format(lep=lep)))
                root_out_files[lep].cd()
                for key in f.GetListOfKeys():
                    h = f.Get(key.GetName())
                    h.Write()
    
    for f in root_out_files:
        f.Close()
    
    print(f"Saved merged CSV to {out_file.name}")


if __name__ == "__main__":
    main()