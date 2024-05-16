#!/usr/bin/env python

'''
Run script for the ranking plots, copied from NPCheck. Used as the executable for tte
condor submit file.
'''
import sys, ROOT

filename = sys.argv[1]
np = sys.argv[2]
fccs = sys.argv[3]
if fccs == "NONE":
    fccs = ""
limit = sys.argv[4]
if limit == "NONE":
    limit = ""
doPostFit = int(sys.argv[5])
doAsimov = int(sys.argv[6])
outname = sys.argv[7]
macroPath = sys.argv[8]

ROOT.gROOT.LoadMacro(macroPath)
ROOT.setup(filename, fccs, limit, doPostFit, doAsimov, outname)
ROOT.runSingle(np, ROOT.getNPUp(np), "_up")
ROOT.runSingle(np, ROOT.getNPDown(np), "_down")
