#!/usr/bin/env python3
'''
@file merge_gpr_condor.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date April 2, 2024 
@brief Merges the CSV files of condor GPR runs

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------


'''

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import shutil


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

    first_line = True
    with open(os.path.join(base_dir, csv_name), 'w') as out_file:
        for dirpath, dirnames, filenames in os.walk(base_dir):
            if dirpath == base_dir: continue
            if csv_name in filenames:
                with open(os.path.join(dirpath, csv_name), 'r') as in_file:
                    for i,line in enumerate(in_file):
                        if first_line:
                            out_file.write(line)
                            first_line = False
                        elif i > 0: # skip header
                            out_file.write(line)
    
    print(f"Saved merged file to {out_file.name}")


if __name__ == "__main__":
    main()