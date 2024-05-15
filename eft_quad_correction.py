#!/usr/bin/env python
'''
@author Rob Les
'''

import ROOT  # type:ignore
import sys, os
from argparse import ArgumentParser


def get_pdf(ws, pdf, operator):
    # Make the new RooSimulateous, aka master pdf
    newPdf = ROOT.RooSimultaneous(pdf)
    replace_old = [f"mu-{operator}-lin", f"mu-{operator}-quad"]
    replace_new = ["muLn", "muSq"]
    getattr(ws, "import")(
        newPdf,
        ROOT.RooFit.RenameVariable(",".join(replace_old), ",".join(replace_new)),
        ROOT.RooFit.RecycleConflictNodes(),
    )
    return newPdf


def getWSInfo(filename):
    workspaceName = ""
    dataName = ""
    modelConfigName = "ModelConfig"

    workspaceFile = ROOT.TFile(filename, "READ")
    for key in workspaceFile.GetListOfKeys():
        if key.GetClassName() == "RooWorkspace":
            ws = workspaceFile.Get(key.GetName())
            if ws:
                print("Using workspace with name %s" % ws.GetName())
                workspaceName = ws.GetName()
                break
    if workspaceName == "":
        print("Couldn't find workspace!")
    possibleDataName = ["obsData", "combData"]
    for dName in possibleDataName:
        data = ws.data(dName)
        if data:
            print("Using data with name %s" % dName)
            dataName = dName
            break
    if data == None:
        print("Couldn't find data!")
    workspaceFile.Close()

    return workspaceName, dataName, modelConfigName


def joinPOI(filename, outname, operator):

    workspaceName, dataName, modelConfigName = getWSInfo(filename)
    workspaceFile = ROOT.TFile(filename, "READ")
    ws = workspaceFile.Get(workspaceName)
    mc = ws.obj(modelConfigName)

    new_ws = ROOT.RooWorkspace(ws.GetName())
    new_mc = ROOT.RooStats.ModelConfig(mc.GetName(), new_ws)
    new_mc.SetGlobalObservables(mc.GetGlobalObservables())
    new_mc.SetNuisanceParameters(mc.GetNuisanceParameters())
    new_mc.SetObservables(mc.GetObservables())
    for _data in ws.allData():
        getattr(new_ws, "import")(_data)
    new_ws.factory(f"mu-{operator}[0,-100,100]")
    new_ws.factory(f"expr::muSq('@0*@0',{{mu-{operator}}})")
    new_ws.factory(f"expr::muLn('@0',{{mu-{operator}}})")

    # Make the new pdf, this is where the real work is
    new_pdf = get_pdf(new_ws, mc.GetPdf(), operator)
    new_mc.SetPdf(new_pdf)
    new_mc.SetParametersOfInterest(new_ws.var(f"mu-{operator}"))
    getattr(new_ws, "import")(new_mc)

    new_ws.writeToFile(outname)


if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument("infile")
    parser.add_argument("--outfile")
    args = parser.parse_args()

    if not os.path.exists(args.infile):
        print("No file found?")
        sys.exit()

    joinPOI(args.infile, args.outfile, 'cw')
