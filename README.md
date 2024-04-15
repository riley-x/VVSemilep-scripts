# VVSemileptonic Postscripts

This repository contains many post-scripts related to the VV Semileptonic analysis, mostly related to the unfolding analysis. It takes as input the histograms produced by the CxAODReader. 

**In general, see the docstrings of the individual files and the --help text of each executable for more (and usually more up-to-date) info.**

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

A master run script [master.py](master.py) runs the full workflow. It can be called by simply supplying the file format of the histogram files. 

```sh
master.py /path/to/hists/hist_{sample}_{lep}-0.root \
    [--channels 1] \ 
    [--output ./output] \
    [--skip-fits] \
    [--skip-gpr] \
    [--condor] \
    [--asimov] 
```

This script takes care of the following tasks, which can all be called individually too (see below).

1. Fitting the ttbar signal strength to 1lep TCR.
2. Fitting the GPR to the event-subtracted MCR.
3. Repeating the above for every systematic variation.
4. Creating response matrices.
5. Performing the profile likelihood unfolding fit.

The main input is a formatter path to the histogram files. These are assumed to follow a systematic naming scheme. Use python-esque formatters `{sample}` and `{lep}` to encode variable fields that will be replaced by the sample name and the lepton channel name. These are all hardcoded in [utils.py](utils.py). The `Sample` class is defined with the filestubs that will be tried for each sample, and the lepton stubs are taken from `FileManager.lepton_channel_names`. Note that `FileManager` implements the file and histogram fetcher and is used throughout. Be careful to not duplicate files; the script will greedily try every possible combination of names and will add them together if multiple are found (good for when you have separate files that need to be added anyways, like different campaigns. But bad if you're not expecting it). Multiple formatters can be passed to `master.py`, but again be careful about accidentally duplicating files.

Outputs are stored to the output directory, which defaults to `./output`. Most importantly are the plots in the `plots` subdirectory. GPR results, which take the longest runtime, are stored in `gpr/gpr_fit_results.csv`. Note that these can be batched by calling `--condor`, which is especially helpful when running systematics. The results of each run can be merged with `merge_gpr_condor.py`. You can then rerun `master.py` by passing `--skip-gpr` to skip running the GPR and just read the CSV instead. This can also be used i.e. if you generate the GPR results without condor but just want to rerun something else. Similarly, `--skip-fits` also skips the PLU fit in addition, and will use the stored fit results in `rf/{lep}lep_{var}.fcc.root`.

### Config

Check [utils.py](utils.py) for some hard coded configuration, especially the binning of each variable and the naming of variables/samples/systematics/histograms. Notably, you may want to run the binning optimizer in the unfolding script below. These need to be hand copied to replace the binning in `get_bins` of `utils.py` to be used by the master script.

Set the `--asimov` flag to indicate running over asimov data. `FileManager` will look for files named `data-asimov` instead of `data` in the `{sample}` field. These files need to first be generated using i.e. `make_asimov.py`, which simply adds the MC channels together.


### Unfolding

You can run [unfolding.py](unfolding.py) to generate plots of the migration matrix, efficiency, acceptance, and fiducial/reco distributions easily from the reader output histograms. These generally should be done on the diboson samples, but works for all samples.

```sh
unfolding.py path/to/reader/output/1lep/hist-Diboson.root diboson 1 
```

Here the `diboson` is the name of the sample as in [utils.py](utils.py) and the `1` is the lepton channel. 
Variables are hardcoded in the file; edit them as needed. You can optionally pass `--optimize 500,3000` for example to run the automatic bin optimizations in the range specified. See hardcoded optimization parameters in `optimize_binning`.

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
- `postfit`: a "dumb" postfit in m(J) that stacks the backgrounds and calculates the MC signal strength using `SR yield - GPR yield`. Just used for diagnostics. 

### ttbar fit

Use [ttbar_fit.py](ttbar_fit.py) to run the ttbar fit to the one-lep TCR. This constrains the ttbar signal strength. See the docstring for more info.



