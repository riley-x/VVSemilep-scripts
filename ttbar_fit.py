#!/usr/bin/env python3
'''
@file ttbar_fit.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date February 21, 2024 
@brief Script for fitting the ttbar/stop backgrounds using the 1lep channel.

------------------------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------------------------

    setupATLAS 
    lsetup "root recommended" 
    lsetup "python centos7-3.9"

Note that this can't be setup at the same time with AnalysisBase or else you get a lot of
conflicts :(

------------------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------------------

Check [utils.Sample] and [utils.Variable] to make sure the hardcoded naming stuctures are
correct.

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------

    ttbar_fit.py filepath/formatter_1.root [...]

This will fetch histogram files using the naming convention supplied in the arguments.
These arguments can include python formatters (using curly braces) for 'lep', which will
be replaced with the lepton channel number, and 'sample', which uses
[utils.Sample.file_stubs]. For example,

    hists/{lep}lep/{sample}.root
    
See [utils.FileManager] for details.
'''

import ROOT
import numpy as np
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from plotting import plot
import utils



##########################################################################################
###                                         RUN                                        ###
##########################################################################################

def hessian(f, x, delta):
    '''
    Calculates the Hessian of [f] at [x] using finite differences of step size [delta].
    '''
    f0 = f(x)
    out = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            xp = np.array(x)
            if i == j:
                xp[i] = x[i] + delta[i]
                val_up = f(xp)
                xp[i] = x[i] - delta[i]
                val_down = f(xp)
                val = val_up + val_down - 2 * f0
                val /= delta[i] * delta[i]
            else:
                # https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470824566.app1 A.7
                xp[i] = x[i] + delta[i]
                xp[j] = x[j] + delta[j]
                val = f(xp)

                xp[i] = x[i] - delta[i]
                val -= f(xp)

                xp[j] = x[j] - delta[j]
                val += f(xp)

                xp[i] = x[i] + delta[i]
                val -= f(xp)

                val /= 4 * delta[i] * delta[j]

            out[i, j] = val
            out[j, i] = val

    return out


def run_fit(file_manager : utils.FileManager):
    from scipy import optimize, stats

    ### Get hists ###
    hist_name = '{sample}_VV1Lep_MergHP_Inclusive_TCR_lvJ_m'
    hists = file_manager.get_hist_all_samples(1, hist_name)

    n_data = plot.integral_user(hists['data'], return_error=True)
    n_ttbar = plot.integral_user(hists['ttbar'], return_error=True)
    n_stop = plot.integral_user(hists['stop'], return_error=True)

    h_else = hists['wjets'].Clone()
    h_else.Add(hists['zjets'])
    h_else.Add(hists['diboson'])
    n_else = plot.integral_user(h_else, return_error=True)

    ### Define NLL form ###
    def nll(params):
        mu_ttbar = params[0]
        mu_stop = params[1]
        gamma_mc = params[2]

        mc_error = (mu_ttbar * n_ttbar[1])**2 + (mu_stop * n_stop[1])**2 + n_else[1]**2
        mc_error = mc_error**0.5

        pred = mu_ttbar * n_ttbar[0] + mu_stop * n_stop[0] + n_else[0] + gamma_mc * mc_error
        out = -stats.poisson.logpmf(n_data[0], pred) \
            - stats.norm.logpdf(mu_stop, loc=1, scale=0.2) \
            - stats.norm.logpdf(gamma_mc)
        return out
        
    ### Minimize ###
    res = optimize.minimize(nll, [1, 1, 0], bounds=[(1e-2, 2), (1e-2, 2), (-5, 5)], method='L-BFGS-B')#, options={'ftol': 1e-15, 'gtol': 1e-15})
    if not res.success:
        plot.error(f'ttbar_fit.py::run_fit() did not succeed:\n{res}')
        raise RuntimeError()

    ### Covariances ###
    # Note we don't use the Scipy covariance which is not too accurate
    # cov = res.hess_inv.todense()
    hess = hessian(nll, res.x, [0.001, 0.001, 0.001])
    cov = np.linalg.inv(hess)
    errs = np.diag(cov) ** 0.5
    cov_norm = cov / errs / errs[:, None]
    out = {
        'mu_ttbar': (res.x[0], errs[0]),
        'mu_stop': (res.x[1], errs[1]),
        'gamma_mc': (res.x[2], errs[2]),
        'cov': cov,
        'cov_norm': cov_norm,
    }
    
    ### Printout ###
    notice_msg = 'ttbar_fit.py::run_fit() fit results:'
    for k,v in out.items():
        if 'cov' in k: continue
        notice_msg += f'\n    {k:10}: {v[0]:7.4f} +- {v[1]:.4f}'
    notice_msg += f'\n    cov:'
    for i in range(len(errs)):
        notice_msg += f'\n        '
        for j in range(len(errs)):
            notice_msg += f'{cov_norm[i][j]:7.4f}  '
    plot.notice(notice_msg)
    
    return out



##########################################################################################
###                                        MAIN                                        ###
##########################################################################################

def parse_args():
    parser = ArgumentParser(
        description="Master run script for doing the unfolded analysis for VVsemilep.", 
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('filepaths', nargs='+')
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
    args = parse_args()
    file_manager = get_files(args.filepaths)

    plot.save_transparent_png = False
    plot.file_formats = ['png', 'pdf']

    run_fit(file_manager)


if __name__ == "__main__":
    main()