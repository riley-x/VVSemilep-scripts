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
###                                         FIT                                        ###
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


def run_fit(
        file_manager : utils.FileManager, 
        mu_stop_0=(1, 0.2),
        variation='nominal',
    ):
    '''
    @param mu_stop_0
        The stop signal strength constraint (mean, error). Set the error to like 1e-5 to
        treat the stop as fixed.
    '''
    from scipy import optimize, stats

    ### Get hists ###
    hists = file_manager.get_hist_all_samples(1, '{sample}_VV1Lep_MergHP_Inclusive_TCR_lvJ_m', variation)

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
        out = -stats.poisson.logpmf(round(n_data[0]), pred) \
            - stats.norm.logpdf(mu_stop, loc=mu_stop_0[0], scale=mu_stop_0[1]) \
            - stats.norm.logpdf(gamma_mc)
        return out
        
    ### Minimize ###
    res = optimize.minimize(nll, [1.0, mu_stop_0[0], 0], bounds=[(1e-2, 2), (1e-2, 2), (-5, 5)], method='L-BFGS-B', options={'eps':1e-10})#, options={'ftol': 1e-15, 'gtol': 1e-15})
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
    notice_msg = f'ttbar_fit.py::run_fit({variation}) fit results:'
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
###                                     Aggregator                                     ###
##########################################################################################

class TtbarSysFitter:
    '''
    Do ttbar fit with systematics. But don't correlate mu_stop_0 or any other systematics
    at this point.
    '''
    def __init__(self, file_manager: utils.FileManager, mu_stop_0=(1, 0.2)):
        self.file_manager = file_manager
        self.vars = {}

        results_nom = run_fit(file_manager, mu_stop_0=(mu_stop_0[0], 1e-5))
        results_stop_up = run_fit(file_manager, mu_stop_0=(mu_stop_0[0] + mu_stop_0[1], 1e-5), variation=utils.variation_mu_stop + utils.variation_up_key)
        results_stop_down = run_fit(file_manager, mu_stop_0=(mu_stop_0[0] - mu_stop_0[1], 1e-5), variation=utils.variation_mu_stop + utils.variation_down_key)

        self._mu_stop_nom = (mu_stop_0[0], 1e-5)
        self.mu_ttbar_nom = results_nom['mu_ttbar']
        self.vars[utils.variation_nom] = self.mu_ttbar_nom[0]
        self.vars[utils.variation_mu_ttbar + utils.variation_up_key] = self.mu_ttbar_nom[0] + self.mu_ttbar_nom[1]
        self.vars[utils.variation_mu_ttbar + utils.variation_down_key] = self.mu_ttbar_nom[0] - self.mu_ttbar_nom[1]
        self.vars[utils.variation_mu_stop + utils.variation_up_key] = results_stop_up['mu_ttbar'][0]
        self.vars[utils.variation_mu_stop + utils.variation_down_key] = results_stop_down['mu_ttbar'][0]

    def get_var(self, variation):
        out = self.vars.get(variation)
        if out is not None: return out
        res = run_fit(self.file_manager, mu_stop_0=self._mu_stop_nom, variation=variation)
        self.vars[variation] = res['mu_ttbar'][0]
        return self.vars[variation]


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