# VVSemileptonic Postscripts

This repository contains many post-scripts related to the VV Semileptonic analysis, mostly related to the unfolding analysis. It takes as input the histograms produced by the CxAODReader.

## Setup

To install, simple call
```sh
git clone --recursive git@github.com:riley-x/VVSemilep-scripts.git
```
Note that this submodules the private [ResonanceFinder](https://gitlab.cern.ch/atlas-phys/exot/dbl/ResonanceFinder) repo. If you don't have access, simple init the `plotting` submodule manually instead of using the `--recursive` flag above.

Setup using
```sh
. setup.sh
```
This does one of two things: if you have access to ResonanceFinder, it sets it up via 
```sh
cd ResonanceFinder
. setup_RF.sh
cd ..
```
which includes all the necessary python packages, and also compiles (initially) ResonanceFinder. 
Otherwise (assuming you're on Tier 3 systems), it will call 
```sh
setupATLAS
lsetup "root recommended"
lsetup "python centos7-3.9"
```
Note that this can't be setup at the same time with AnalysisBase because it is stuck on python2 still. If you're running locally, you'll need to `pip install` any dependencies. 

## Run

A master run script [master.py](master.py) runs the full workflow. See the docstring of that file for more information. It takes care of the following tasks, which can all be called individually too (see below).

1. Fitting the ttbar signal strength to 1lep TCR.
2. Fitting the GPR to the event-subtracted MCR.
3. Repeating the above for every systematic variation.
4. Creating response matrices.
5. Performing the profile likelihood unfolding fit.

The script relies on systematic histogram naming. See the docstring of the file and [utils.py](utils.py) for more info.


### Unfolding

You can run [unfolding.py](unfolding.py) to generate plots of the migration matrix, efficiency, acceptance, and fiducial/reco distributions easily from the reader output histograms. These generally should be done on the diboson samples, but works for all samples.

```sh
unfolding.py path/to/reader/output/1lep/hist-Diboson.root SMVV 1
```

Here the `SMVV` is the name of the sample as stored in the root file, and the `1` is the lepton channel. 
Some variables and binnings are hardcoded in the file; edit them as needed.

This script uses the histograms stored via the `fillUnfHistograms` of each `AnalysisReader` class. These generally leverage `AnalysisReaderVV::fillUnfoldingMatrix` to fill a single matrix (for each variable) containing all the necessary info for PLU. The matrix is a 2d reco vs fiducial accumulator. Events that fail 
either the fiducial or reco selections are encoded in the respective underflows. 
This has the advantage that if you rebin the matrix to trim the boundaries, the
efficiency and accuracy are automatically updated. 
Note then that you can efficiently obtain the ingredients for PLU as follows:

- Migration matrix: from the non-underflow entries (must normalize each column)
- Fiducial distribution: simply call ProjectionX()
- Detector distribution: simply call ProjectionY()
- Efficiency: divide ProjectionX(1, n) / ProjectionX()
- Accuracy: divide ProjectionY(1, n) / ProjectionY()


This script also creates a file with the response matrix broken down by bin, for inputting into ResonanceFinder.
The histograms have names `ResponseMatrix_{var}_fid{bin}`, for each variable and fiducial bin. 
These should be treated as signal samples, and are normalized such that the signal strength is exactly the unfolded fiducial event count for the respective bin. 

### GPR

To run the data-driven background estimate, use [gpr.py](gpr.py). Please see the docstring of that file for more details.

```sh
gpr.py --lepton 1 --var vv_m [FILESPEC]
```
        
This runs the GPR fit to a single variable in one channel. It uses as inputs the `{var}__v__fatjet_m`
histograms in the reader, which are filled with the inclusive SR+MCR. 
For each sample, it will run a fit for each bin specified in `get_bins_y`. In general, see the `FitConfig` class for some hardcodes.

The fit runs the full contour scan to obtain the marginalized posterior. Results are saved into a file
`gpr_fit_results.csv` containing both a 2sigma contour scan and the simple MMLE fit, and a ROOT histogram in
`gpr_{lep}_{var}_vjets_yield.root` which can be input into ResonanceFinder. Note that the histogram file
can be created from the CSV directly without rerunning the fits using the `--fromCsvOnly` option, if you need to edit or rerun a specific bin.

The script will generate a plot named `gpr_{lep}_{var}_summary` in both png and pdf formats containing a summary distribution of the fits. For each bin fit, will also generate the following plots:

- `nlml_cont`: contour lines of the NLML space as a function of the two hyperparameters.
- `yields`: fitted SR yields in the same space.
- `fit_opt`: posterior prediction using the MMLE fit
- `fit_m1s`/`fit_p1s`: posterior predictions using +-1 sigma in the hyperparameter space fits
- `fit_pm1s`: the above 3 posterior means superimposed onto the same plot
- `p_intr`: the marginalized posterior yield distribution

### ttbar fit

Use [ttbar_fit.py](ttbar_fit.py) to run the ttbar fit to the one-lep TCR. This constrains the ttbar signal strength. See the docstring for more info.



