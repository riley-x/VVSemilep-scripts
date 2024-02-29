#!/usr/bin/env python3

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

def hadd(target, *sources):
    print('hadd', target, *sources)

    src = []
    for source in sources:
        src.extend(glob.glob(source))
    args = ['hadd', '-k', target, *src]

    res = subprocess.run(args) 
    res.check_returncode()


for lep in [1]:
    for campaign in ['a']:
        if campaign == 'a':
            campaign_samples = samples + ['data15', 'data16']
        elif campaign == 'd':
            campaign_samples = samples + ['data17']
        else:
            campaign_samples = samples + ['data18']

        for sample in campaign_samples:
            sources = f'{base_dir}/mc16{campaign}/{lep_dirs[lep]}/{sample}/*/*'
            hadd(f'hists/{lep}lep_{sample}_{campaign}.{stub_name}.hists', sources)


            # args = ['hadd', '-k', f'{out_base}.root', f'{out_base}_*.root'] 


            # dir_contents = os.scandir(hist_dir)

            # out_base = f'{lep}lep_{sample}_{campaign}.{stub_name}.hists'
            # for i,f in enumerate(dir_contents):
            #     args = ['hadd', '-k', f'{out_base}_{i}.root', f'{f.path}/*'] # the -k skips empty/corrupt files
            #     print(args)
            #     # res = subprocess.run(args) 
            #     # res.check_returncode()
            
            # args = ['hadd', '-k', f'{out_base}.root', f'{out_base}_*.root'] 
            # print(args)
