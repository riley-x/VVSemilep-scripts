#!/usr/bin/env python3
'''
@file diboson_fit.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date February 22, 2024 
@brief Functions for fitting the diboson yield directly, bin-by-bin
'''

import ROOT # type: ignore
import numpy as np
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from plotting import plot
import utils
import ttbar_fit
import gpr
import master



##########################################################################################
###                                         FIT                                        ###
##########################################################################################


def run_fit(
        config : master.ChannelConfig,
        variable : utils.Variable,
        bin : tuple[float, float],
        gpr_mu_corr : float,
    ):
    from scipy import optimize, stats

    ### Setup ###
    hist_name = '{sample}_VV{lep}_MergHP_Inclusive_SR_' + utils.generic_var_to_lep(variable, config.lepton_channel).name
    variations = utils.variations_custom + utils.variations_hist
    gpr_csv_args = dict(
        lep=config.lepton_channel,
        vary=variable.name,
        fitter='rbf_marg_post',
        bin=bin,
        unscale_width=True,
    )

    ### Get nominal yields ###
    def _subr_get_nominal(asimov : bool):
        hists = config.file_manager.get_hist_all_samples(config.lepton_channel, hist_name, utils.variation_nom)
        
        ### Data and diboson ###
        n_data_nom = plot.integral_user(hists['data'], bin)
        n_data_nom = round(n_data_nom) # must be int for poisson
        n_diboson_nom = plot.integral_user(hists['diboson'], bin, return_error=True)
        
        ### MC top backgrounds (sum) ###
        mu_ttbar_nom = config.ttbar_fitter.mu_ttbar_nom
        n_ttbar_nom = plot.integral_user(hists['ttbar'], bin, return_error=True)
        n_stop_nom = plot.integral_user(hists['stop'], bin, return_error=True)
        val = mu_ttbar_nom[0] * n_ttbar_nom[0] + config.mu_stop[0] * n_stop_nom[0]
        err = ((mu_ttbar_nom[0] * n_ttbar_nom[1])**2 + (config.mu_stop[0] * n_stop_nom[1])**2)**0.5
        n_mc_nom = (val, err)

        ### MC V+jets for comparison ###
        n_wjets_nom = plot.integral_user(hists['wjets'], bin, return_error=True)
        n_zjets_nom = plot.integral_user(hists['zjets'], bin, return_error=True)

        ### GPR ###
        n_gpr_nom = config.gpr_results.get_entry(**gpr_csv_args, variation='nominal')
        n_gpr_nom = (n_gpr_nom[0], (n_gpr_nom[1] + n_gpr_nom[2]) / 2)

        ### Nominal signal strength ###
        if asimov:
            n_data_nom = round(n_mc_nom[0] + n_diboson_nom[0] + n_gpr_nom[0]) # must be int for poisson
        
        numerator = n_data_nom - n_mc_nom[0] - n_gpr_nom[0]
        numerator_err = n_data_nom + n_mc_nom[1]**2 + n_gpr_nom[1]**2
        mu_diboson_err = numerator_err / numerator**2 + (n_diboson_nom[1] / n_diboson_nom[0])**2
        mu_diboson_val = numerator / n_diboson_nom[0]
        mu_diboson_nom = (mu_diboson_val, mu_diboson_val * mu_diboson_err**0.5)

        print(f'diboson_fit.py::run_fit({variable}, {bin}):')
        print(f'    Nominal:')
        print('    ' + '-' * 33)
        print(f'    {"Data":10}: {n_data_nom:10.2f}')
        print('    ' + '-' * 33)
        print(f'    {"ttbar":10}: {mu_ttbar_nom[0] * n_ttbar_nom[0]:10.2f} {mu_ttbar_nom[0] * n_ttbar_nom[1]:10.2f}')
        print(f'    {"stop":10}: {config.mu_stop[0] * n_stop_nom[0]:10.2f} {config.mu_stop[0] * n_stop_nom[1]:10.2f}')
        print(f'    {"gpr":10}: {n_gpr_nom[0]:10.2f} {n_gpr_nom[1]:10.2f}')
        print(f'    {"wjets":10}: {n_wjets_nom[0]:10.2f} {n_wjets_nom[1]:10.2f}')
        print(f'    {"zjets":10}: {n_zjets_nom[0]:10.2f} {n_zjets_nom[1]:10.2f}')
        print('    ' + '-' * 33)
        print(f'    {"diff":10}: {numerator:10.2f} {numerator_err**0.5:10.2f}')
        print(f'    {"diboson":10}: {n_diboson_nom[0]:10.2f} {n_diboson_nom[1]:10.2f}')
        print('    ' + '-' * 33)
        print(f'    {"mu":10}: {mu_diboson_nom[0]:10.2f} {mu_diboson_nom[1]:10.2f}')

        print('\n')
        print(f'    {"Variation":20}: {"top MCs":>10} {"GPR":>10}')
        print('    ' + '-' * 43)
        print(f'    {"nominal_val":20}: {n_mc_nom[0]:10.2f} {n_gpr_nom[0]:10.2f}')
        print(f'    {"nominal_err":20}: {n_mc_nom[1]:10.2f} {n_gpr_nom[1]:10.2f}')

        return n_data_nom, n_diboson_nom, n_mc_nom, n_gpr_nom, mu_diboson_nom
    n_data_nom, n_diboson_nom, n_mc_nom, n_gpr_nom, mu_diboson_nom = _subr_get_nominal(asimov=False)

    ### Get variation errors ###
    var_sigmas = {
        'gamma_mc': n_mc_nom[1],
        'gamma_gpr': n_gpr_nom[1],
    }
    var_index = { # mu_diboson is 0
        'gamma_mc': 1,
        'gamma_gpr': 2,
    }
    index = 3
    for variation_base in variations:
        mc_err = 0
        gpr_err = 0
        for updown in ['1up', '1down']:
            variation_updown = f'{variation_base}__{updown}'

            ### Get MC background total (ttbar + stop) ###
            var_name = utils.hist_name_variation(hist_name, variation_updown)
            h_ttbar = config.file_manager.get_hist(config.lepton_channel, utils.Sample.ttbar, var_name, variation_updown)
            h_stop = config.file_manager.get_hist(config.lepton_channel, utils.Sample.stop, var_name, variation_updown)
            n_ttbar = plot.integral_user(h_ttbar, bin)
            n_stop = plot.integral_user(h_stop, bin)

            ### Get signal strengths ###
            mu_ttbar = config.ttbar_fitter.get_var(variation_updown)
            if variation_base == 'mu-stop':
                if updown == 'up':
                    mu_stop_1 = config.mu_stop[0] + config.mu_stop[1]
                else:
                    mu_stop_1 = config.mu_stop[0] - config.mu_stop[1]
            else:
                mu_stop_1 = config.mu_stop[0]

            ### Get diff ###
            val = n_ttbar * mu_ttbar + n_stop * mu_stop_1
            mc_err += (val - n_mc_nom[0]) * (1 if updown == 'up' else -1)

            ### Get GPR err ###
            val = config.gpr_results.get_entry(
                variation=variation_updown,
                **gpr_csv_args,
            )[0]
            gpr_err += (val - n_gpr_nom[0]) * (1 if updown == 'up' else -1)
        
        mc_err /= 2
        gpr_err /= 2
        var_sigmas[f'gamma_{variation_base}'] = mc_err + gpr_err
        var_index[f'gamma_{variation_base}'] = index
        index += 1
        print(f'    {variation_base:20}: {mc_err:10.2f} {gpr_err:10.2f}')

    ### Define NLL form ###
    def nll(params):
        mu_diboson = params[0]
        pred = mu_diboson * n_diboson_nom[0] + n_mc_nom[0] + n_gpr_nom[0]
        pred += gpr_mu_corr * n_gpr_nom[0] * (1 - mu_diboson) # signal contamination correction
        nll_val = 0
        for var,i in var_index.items():
            pred += params[i] * var_sigmas[var]
            nll_val -= stats.norm.logpdf(params[i])

        nll_val -= stats.poisson.logpmf(n_data_nom, pred)
        return nll_val
    
    ### Minimize ###
    n_params = len(var_index) + 1
    params = np.zeros(n_params)
    params[0] = 1
    bounds = [(0, 5)] + [(-4, 4)] * len(var_index)

    res = optimize.minimize(nll, params, bounds=bounds, method='L-BFGS-B')#, options={'ftol': 1e-15, 'gtol': 1e-15})
    if not res.success:
        plot.error(f'diboson_fit.py::run_fit() did not succeed:\n{res}')
        raise RuntimeError()

    ### Covariances ###
    # Note we don't use the Scipy covariance which is not too accurate
    # cov = res.hess_inv.todense()
    hess = ttbar_fit.hessian(nll, res.x, [0.001] * n_params)
    cov = np.linalg.inv(hess)
    errs = np.diag(cov) ** 0.5
    cov_norm = cov / errs / errs[:, None]
    out = {
        'mu-diboson-nom': mu_diboson_nom,
        'mu-diboson': (res.x[0], errs[0]),
        'diboson-yield': (res.x[0] * n_diboson_nom[0], errs[0] * n_diboson_nom[0]),
        'diboson-yield-mc-statonly': n_diboson_nom,
        'cov': cov,
        'cov_norm': cov_norm,
    }
    for var,i in var_index.items():
        out[var] = (res.x[i], errs[i])
    
    ### Printout ###
    max_key_length = max(len(k) for k in out.keys())
    notice_msg = f'\n    Fit results:\n    ' + '-' * (21 + max_key_length)
    for k,v in out.items():
        if 'cov' in k: continue
        notice_msg += f'\n    {k:{max_key_length}}: {v[0]:8.4f} +- {v[1]:7.4f}'
    notice_msg += f'\n    cov:'
    for i in range(len(errs)):
        notice_msg += f'\n        '
        for j in range(len(errs)):
            notice_msg += f'{cov_norm[i][j]:7.4f}  '
    print(notice_msg, '\n')

    return out
