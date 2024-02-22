#!/usr/bin/env python3
'''
@file diboson_fit.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date February 22, 2024 
@brief Script for fitting the diboson yield using saved GPR fit results

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

This script relies on the saved GPR fit results, which should be located at
`{output}/gpr/gpr_fit_results.csv`.

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------

    diboson_fit.py filepath/formatter_1.root [...]

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
import ttbar_fit
import gpr



##########################################################################################
###                                         FIT                                        ###
##########################################################################################


def run_fit(
        file_manager : utils.FileManager, 
        gpr_results : gpr.FitResults,
        ttbar_fitter : ttbar_fit.TtbarSysFitter,
        lepton_channel : int, 
        var : utils.Variable,
        bin : tuple[float, float],
        mu_stop : tuple[float, float],
    ):
    '''
    '''
    from scipy import optimize, stats

    ### Setup ###
    hist_name = '{sample}_VV1Lep_MergHP_Inclusive_SR_' + utils.generic_var_to_lep(var, lepton_channel).name
    variations = [
        'mu-ttbar',
        'mu-stop',
    ]
    gpr_csv_args = dict(
        lep=lepton_channel,
        vary=var.name,
        fitter='rbf_marg_post',
        bin=bin,
    )

    ### Get nominal yields ###
    def _subr_get_nominal():
        nom_name = utils.hist_name_variation(hist_name, 'nominal')
        hists = file_manager.get_hist_all_samples(lepton_channel, nom_name)
        
        ### Data and diboson (always nominal) ###
        n_data_nom = plot.integral_user(hists['data'], bin)
        n_diboson_nom = plot.integral_user(hists['diboson'], bin)
        
        ### MC top backgrounds nominal (sum) ###
        mu_ttbar_nom = ttbar_fitter.mu_ttbar_nom
        n_ttbar_nom = plot.integral_user(hists['ttbar'], bin, return_error=True)
        n_stop_nom = plot.integral_user(hists['stop'], bin, return_error=True)
        val = mu_ttbar_nom[0] * n_ttbar_nom[0] + mu_stop[0] * n_stop_nom[0]
        err = ((mu_ttbar_nom[0] * n_ttbar_nom[1])**2 + (mu_stop[0] * n_stop_nom[1])**2)**0.5
        n_mc_nom = (val, err)

        ### GPR nominal ###
        n_gpr_nom = gpr_results.get_entry(**gpr_csv_args, variation='nominal')
        n_gpr_nom = (n_gpr_nom[0], (n_gpr_nom[1] + n_gpr_nom[2]) / 2)

        print(f'    {"Variation":20}: {"top MCs":>10} {"GPR":>10}')
        print('    ' + '-' * 43)
        print(f'    {"nominal_val":20}: {n_mc_nom[0]:10.2f} {n_gpr_nom[0]:10.2f}')
        print(f'    {"nominal_err":20}: {n_mc_nom[1]:10.2f} {n_gpr_nom[1]:10.2f}')

        return n_data_nom, n_diboson_nom, n_mc_nom, n_gpr_nom
    n_data_nom, n_diboson_nom, n_mc_nom, n_gpr_nom = _subr_get_nominal()
    
    ### Get variation errors ###
    var_sigmas_gpr = {}
    var_sigmas_mc = {}
    for variation_base in variations:
        mc_err = 0
        gpr_err = 0
        for updown in ['up', 'down']:
            variation_updown = f'{variation_base}_{updown}'

            ### Get MC background total (ttbar + stop) ###
            var_name = utils.hist_name_variation(hist_name, variation_updown)
            h_ttbar = file_manager.get_hist(lepton_channel, utils.Sample.ttbar, var_name)
            h_stop = file_manager.get_hist(lepton_channel, utils.Sample.stop, var_name)
            n_ttbar = plot.integral_user(h_ttbar, bin)
            n_stop = plot.integral_user(h_stop, bin)

            ### Get signal strengths ###
            mu_ttbar = ttbar_fitter.get_var(variation_updown)
            if variation_base == 'mu-stop':
                if updown == 'up':
                    mu_stop_1 = mu_stop[0] + mu_stop[1]
                else:
                    mu_stop_1 = mu_stop[0] - mu_stop[1]
            else:
                mu_stop_1 = mu_stop[0]

            ### Get diff ###
            val = n_ttbar * mu_ttbar + n_stop * mu_stop_1
            mc_err += abs(val - n_mc_nom[0])

            ### Get GPR err ###
            val = gpr_results.get_entry(
                variation=variation_updown,
                **gpr_csv_args,
            )[0]
            gpr_err += abs(val - n_gpr_nom[0])
        
        mc_err /= 2
        gpr_err /= 2
        var_sigmas_mc[variation_base] = mc_err
        var_sigmas_gpr[variation_base] = gpr_err
        print(f'    {variation_base:20}: {mc_err:10.2f} {gpr_err:10.2f}')
    return

    ### Define NLL form ###
    def nll(params):
        mu_ttbar = params[0]
        mu_stop = params[1]
        gamma_mc = params[2]

        mc_error = (mu_ttbar * n_ttbar[1])**2 + (mu_stop * n_stop[1])**2 + n_else[1]**2
        mc_error = mc_error**0.5

        pred = mu_ttbar * n_ttbar[0] + mu_stop * n_stop[0] + n_else[0] + gamma_mc * mc_error
        out = -stats.poisson.logpmf(n_data[0], pred) \
            - stats.norm.logpdf(mu_stop, loc=mu_stop_0[0], scale=mu_stop_0[1]) \
            - stats.norm.logpdf(gamma_mc)
        return out
            
    ### Minimize ###
    res = optimize.minimize(nll, [1, mu_stop_0[0], 0], bounds=[(1e-2, 2), (1e-2, 2), (-5, 5)], method='L-BFGS-B')#, options={'ftol': 1e-15, 'gtol': 1e-15})
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
    notice_msg = f'ttbar_fit.py::run_fit({variation_base}) fit results:'
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
