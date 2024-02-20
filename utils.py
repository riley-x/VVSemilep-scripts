'''
@file variable.py
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch
@date February 7, 2024
@brief Variable naming and other utilities 
'''
from __future__ import annotations
from typing import Union
import ctypes

from plotting import plot
import ROOT

#########################################################################################
###                                     Varaibles                                     ###
#########################################################################################
class Variable:
    '''
    @property name
        The base name that uniquely identifies the variable. This should match the 
        histogram naming conventions.
    @property title
        A TLatex version of the variable that is used in title text.
    @property unit
        The name of the unit of the variable. Again this should match the histogram 
        outputs.
    @property rebin
        A default rebinning option, see [plot.rebin].
    '''
    def __init__(self, name, title, unit, rebin=None, **plot_opts):
        '''
        @param plot_opts
            Any additional default opts to use during plotting, assuming an event
            distribution is being plotted against this variable on the x axis. These can
            be retrieved with [x_plot_opts].
        '''
        self.name = name
        self.title = title
        self.unit = unit
        self.rebin = rebin
        self._plot_opts = plot_opts

    def __repr__(self) -> str:
        return f'Variable({self.name})'
    
    def __format__(self, __format_spec: str) -> str:
        if __format_spec:
            if self.unit:
                return f'{self.title} [{self.unit}]'
            else:
                return f'{self.title}'
        else:
            return self.name
    
    def x_plot_opts(self):
        return {
            'xtitle': f'{self.title} [{self.unit}]',
            **self._plot_opts
        }
    

Variable.vv_m = Variable(name="vv_m", title="m(VV)", unit="GeV")
Variable.vhad_pt = Variable(name="vhad_pt", title="p_{T}(J)", unit="GeV")

Variable.llJ_m = Variable(name="llJ_m", title="m(llJ)", unit="GeV", rebin=100, logy=True)
Variable.lvJ_m = Variable(name="lvJ_m", title="m(l#nuJ)", unit="GeV", rebin=100, logy=True)
Variable.vvJ_m = Variable(name="vvJ_m", title="m(#nu#nuJ)", unit="GeV", rebin=100, logy=True)

Variable.fatjet_m = Variable(name="fatjet_m", title="m(J)", unit="GeV")



#########################################################################################
###                                      Samples                                      ###
#########################################################################################
class Sample:
    '''
    A class that represents a single sample.

    @property name
        A unique name that identifies the sample. This is used to save output files, etc.
    @property title
        A TLatex version of the name that is used in title text.
    @property file_stubs
        A list of the possible naming stubs used in the histogram files for this sample.
    @property hist_keys
        A list of the naming stubs used in the histogram names for this sample. 
    '''
    ### Static members ###
    # List here for linting, initialized later below
    wjets = None
    zjets = None
    ttbar = None
    stop = None
    diboson = None
    data =  None

    def __init__(self, name : str, title : str, file_stubs : list[str], reader_keys : list[str]):
        self.name = name
        self.title = title
        self.file_stubs = file_stubs
        self.hist_keys = reader_keys
    
    def __format__(self, __format_spec: str) -> str:
        if __format_spec:
            return f'{self.title}'
        else:
            return self.name
    
    @staticmethod
    def list_predefined():
        return [Sample.wjets, Sample.zjets, Sample.ttbar, Sample.stop, Sample.diboson, Sample.data]

    @staticmethod
    def parse(name) -> Sample:
        samples = Sample.list_predefined()
        for x in samples:
            if x.name == name:
                return x
        raise RuntimeError(f"Sample.parse() couldn't parse {name}")


Sample.wjets = Sample('wjets', 'W+jets', ['Wjets_Sherpa2211'], ['WLL', 'WHL', 'WHH'])
Sample.zjets = Sample('zjets', 'Z+jets', ['Zjets_Sherpa2211'], ['ZLL', 'ZHL', 'ZHH'])
Sample.ttbar = Sample('ttbar', 't#bar{t}', ['ttbar'], ['ttbar'])
Sample.stop = Sample('stop', 'single top', ['stop'], ['stop'])
Sample.diboson = Sample('diboson', 'diboson', ['diboson_Sherpa2211', 'Diboson_Sh2211'], ['SMVV'])
Sample.data = Sample('data', 'data', ['data', 'data15', 'data16', 'data17', 'data18'], ['data'])


#########################################################################################
###                                    File Access                                    ###
#########################################################################################

def get_hist(tfile, hist_name):
    '''
    Retrieves [hist_name] from [tfile] with error checking.
    '''
    h = tfile.Get(hist_name)
    if not h or h.ClassName() == 'TObject':
        raise RuntimeError(f"Couldn't retrieve histogram {hist_name} from {tfile}")
    return h


def get_hists_sum(tfile, hist_names):
    '''
    Retrieves the histograms specified in [hist_names] from [tfile] and returns their sum.
    Will skip missing histograms. 
    '''
    h_sum = None
    for name in hist_names:
        h2 = tfile.Get(name)
        if not h2 or h2.ClassName() == 'TObject':
            plot.warning(f"Couldn't retrieve histogram {name} from {tfile}.")
        elif h_sum is None:
            h_sum = h2
        else:
            h_sum.Add(h2)
    if h_sum is None:
        raise RuntimeError(f"get_hists_sum() unable to retrieve histograms from {tfile}.")
    return h_sum
    
# If you just pass a python 0 to the ROOT char*, I think it gets interpreted as '0' or something.
_null_char_p = ctypes.c_char_p(0)


class FileManager:
    '''
    This class helps manage retrieving histograms for many samples from a multitude of 
    files. In general, each sample may be split across multiple files and the histograms
    divided by subsamples. This class will take care of adding them all together.

    Make sure though that there are not redundant files that accidentally duplicate the 
    data!
    '''

    def __init__(self, samples : list[Sample], file_path_formats : list[str], lepton_channels = [0, 1, 2]) -> None:
        '''
        Tries to open every file given a set of samples and path formats. 

        @param file_path_formats
            Pass any number file paths that can optionally contain fields "{lep}" which
            will be replaced with each channel in [lepton_channels] and "{sample}" which
            will be replaced with [Sample.file_stubs] for each sample.
        '''
        self.samples = { sample.name : sample for sample in samples }
        self.files = { 
            (lep, sample.name) : self._get_sample_files(lep, sample, file_path_formats) 
            for lep in lepton_channels
            for sample in samples 
        }


    def _get_sample_files(self, lep : int, sample : Sample, file_path_formats: list[str]) -> list[ROOT.TFile]:
        files = []
        for path in file_path_formats:
            for stub in sample.file_stubs:
                formatted_path = path.format(lep=lep, sample=stub)
                ROOT.gSystem.RedirectOutput("/dev/null") # mute TFile error messages
                try:
                    files.append(ROOT.TFile(formatted_path))
                except OSError as e:
                    pass
                ROOT.gSystem.RedirectOutput(_null_char_p, _null_char_p)
        
        if not files:
            plot.warning(f'FileManager() unable to find files for {sample} in the {lep}-lep channel.')
        return files
    

    def get_hist(self, lep : int, sample : str, hist_name_format : str) -> Union[ROOT.TH1F, None]:
        '''
        Retrieves a single histogram given a sample and name format. Histograms from every
        sample file in [self.files] using every key in [sample.hist_keys] will be fetched
        and added together.

        @param hist_name_format
            The name of the histogram, which can optionally contain formatters "{lep}"
            which will be replaced with [lep] and "{sample}" which will be replaced with
            [sample.hist_keys].
        '''
        files = self.files[(lep, sample)]
        sample_obj = self.samples[sample]

        h_out = None
        for key in sample_obj.hist_keys:
            name = hist_name_format.format(lep=lep, sample=key)
            for file in files:
                h = file.Get(name)
                if not h or h.ClassName() == 'TObject':
                    pass
                elif h_out is None:
                    h_out = h
                else:
                    h_out.Add(h)
        return h_out