#!/usr/bin/env python3
'''
@file master.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date February 29, 2024 
@brief Quick script to merge CxAODReader output histograms in the shared eos space.
'''

import subprocess
import os
import glob

stub_name='Feb22_ANN'
base_dir=f'/eos/atlas/atlascerngroupdisk/phys-hdbs/dbl/VVsemilep2nd/Histograms_{stub_name}'
lep_dirs = [
    'HIGG5D1',
    'HIGG5D2',
    'HIGG2D4',
]
samples = [
    'EFT',
    'ttbar',
    'stop',
    'Diboson_Sh2211',
    'Wjets_Sherpa2211',
    'Zjets_Sherpa2211',
]

processes = []

def msg(x):
    '''Prints a green message to the terminal.'''
    print('\033[92m' + x + '\033[0m')

def hadd(target, *sources):
    src = []
    for source in sources:
        src.extend(glob.glob(source))
    args = ['hadd', '-k', target, *src] # the -k skips empty/corrupt files

    res = subprocess.Popen(args)
    res.target = target

    msg('Launched ' + ' '.join(['hadd', target, *sources]))
    processes.append(res)


for lep in [1]:
    for campaign in ['d', 'e']:
        if campaign == 'a':
            campaign_samples = samples + ['data15', 'data16']
        elif campaign == 'd':
            campaign_samples = samples + ['data17']
        else:
            campaign_samples = samples + ['data18']

        for sample in campaign_samples:
            sources = f'{base_dir}/mc16{campaign}/{lep_dirs[lep]}/{sample}/*/*'
            hadd(f'hists/{lep}lep_{sample}_{campaign}.{stub_name}.hists', sources)

            # dir_contents = os.scandir(hist_dir)
            # out_base = f'{lep}lep_{sample}_{campaign}.{stub_name}.hists'
            # for i,f in enumerate(dir_contents):
            #     args = ['hadd', '-k', f'{out_base}_{i}.root', f'{f.path}/*'] # the -k skips empty/corrupt files
            # args = ['hadd', '-k', f'{out_base}.root', f'{out_base}_*.root'] 


for x in processes:
    x.wait()
    msg(f'Completed {x.target}')