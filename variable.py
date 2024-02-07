'''
@file variable.py
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch
@date February 7, 2024
@brief Variable naming utility class 
'''
class Variable:
    def __init__(self, name, xtitle, unit):
        self.name = name
        self.xtitle = xtitle
        self.unit = unit
    
    def __format__(self, __format_spec: str) -> str:
        if __format_spec:
            if self.unit:
                return f'{self.xtitle} [{self.unit}]'
            else:
                return f'{self.xtitle}'
        else:
            return self.name
    
vv_m = Variable(name="vv_m", xtitle="m(VV)", unit="GeV")

