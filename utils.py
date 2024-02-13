'''
@file variable.py
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch
@date February 7, 2024
@brief Variable naming and other utilities 
'''
from plotting import plot

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
    '''
    def __init__(self, name, title, unit):
        self.name = name
        self.title = title
        self.unit = unit
    
    def __format__(self, __format_spec: str) -> str:
        if __format_spec:
            if self.unit:
                return f'{self.title} [{self.unit}]'
            else:
                return f'{self.title}'
        else:
            return self.name
    

vv_m = Variable(name="vv_m", title="m(VV)", unit="GeV")
vhad_pt = Variable(name="vhad_pt", title="p_{T}(J)", unit="GeV")

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
            plot.warning(f"Couldn't retrieve histogram {name} from {tfile}")
        elif h_sum is None:
            h_sum = h2
        else:
            h_sum.Add(h2)
    if h_sum is None:
        raise RuntimeError(f"get_hists_sum() unable to retrieve histograms from {tfile}")
    return h_sum
    