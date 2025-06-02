#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# Neuromorphic Nanowire Network Simulation
Author: Yinhao Xu <yinhao.xu@sydney.edu.au>
"""

import numpy as np 
from tqdm.autonotebook import tqdm as pbar

class NanowireNetwork:
    """
    Nanowire Network Class
    """
    def __init__(self, 
                adjmtx,
                vset:float      = 0.01,
                vreset:float    = 0.005,
                ron:float       = 12.9e3,
                roff:float      = 12.9e6,
                fluxcrit:float  = 0.01,
                fluxmax:float   = 0.015,
                boost:float     = 10.0,
                eta:float       = 1.0
                 ) -> None:
        self.adjmtx     = adjmtx
        self.vset       = vset 
        self.vreset     = vreset 
        self.ron        = ron 
        self.roff       = roff 
        self.fluxcrit   = fluxcrit 
        self.fluxmax    = fluxmax 
        self.boost      = boost 
        self.eta        = eta
        self.nodenum:int= self.adjmtx.shape[0]
        self.edgenum:int= np.sum(np.triu(self.adjmtx)!=0)
        self.edgelist   = np.argwhere(np.triu(self.adjmtx))
        self.flux       = np.zeros(self.edgenum) #Assuming 0 flux initialisation
        self.conductance= np.zeros(self.edgenum) #Thus 0 cond initialisation

    def update_conductance(self):
        """
        Conductance updated with the tunnelling memristor model.
        This is an exact copy from the original Python implementation found in
        https://github.com/rzhu40/ASN_simulation
        """
        phi = 0.81
        C0 = 10.19
        J1 = 0.0000471307
        A = 0.17
        d = (self.fluxcrit - abs(self.flux))*5/self.fluxcrit
        d[d<0] = 0 
        rt = 2/A * d**2 / phi**0.5 * np.exp(C0*phi**2 * d)/J1
        self.conductance = 1/(rt + self.ron) +  1/self.roff
        return self.conductance
    
    def update_flux(self, edge_voltages, dt:float):
        """
        Updating the flux linkage of every memristive edge
        """
        dflux = (abs(edge_voltages) > self.vset) *\
                (abs(edge_voltages) - self.vset) *\
                np.sign(edge_voltages) 
        dflux = dflux + \
                (abs(edge_voltages) < self.vreset) *\
                (abs(edge_voltages) - self.vreset) *\
                np.sign(self.flux) * self.boost
        self.flux = self.flux + dt*dflux*self.eta
        self.flux[abs(self.flux) > self.fluxmax] = \
                np.sign(self.flux[abs(self.flux) > self.fluxmax])*self.fluxmax
        return self.flux
    
def get_node_voltages(nwn, signal):
    n = nwn.nodenum
    lhs = nwn.Gmtx 
    rhs = nwn.rhs
    lhs[nwn.edgelist[:,0], nwn.edgelist[:,1]] = -nwn.conductance 
    lhs[nwn.edgelist[:,1], nwn.edgelist[:,0]] = -nwn.conductance 
    lhs[range(n), range(n)] = 0 #needs to be here for correct sum below
    lhs[range(n), range(n)] = -np.sum(lhs[:n,:n], axis=0)
    rhs[n:] = signal
    sol = np.linalg.solve(lhs, rhs)
    return sol[:n]

def neuro_sim(
        nwn,
        electrodes:list,
        electrode_signals,
        dt:float    = 0.05,
        steps:int   = 10,
        sig_augment = None,
        return_flux:bool = False,
        disable_pbar:bool = False,
    ):
    """
    Input:
        nwn_params          (dict):     NanowireNetwork object
        electrodes          (list):     list of electrode node indices for the input and drain electrodes
        electrode_signals   (ndarray):  array of input signals
        dt                  (float):    timestep size of simulation
        steps               (int):      how many steps to simulate
        return_flux         (bool):     whether to return the edge flux vals
        disable_pbar        (bool):     whether to disable the progress bar
    Returns:
        (ndarray): all node voltage signals
    """
    n = nwn.nodenum
    m = len(electrodes)
    node_voltages = np.zeros((steps, n))
    edge_voltages = np.zeros(nwn.edgenum)
    if return_flux:
        edge_fluxes = np.zeros((steps, nwn.edgenum))
    update_signal = sig_augment is not None

    nwn.Gmtx = np.zeros((n+m, n+m))
    nwn.rhs = np.zeros(n+m)
    for i, this_elec in enumerate(electrodes):
        nwn.Gmtx[n+i, this_elec] = 1
        nwn.Gmtx[this_elec, n+i] = 1
    for t in pbar(
        range(steps), desc='|NWN Sim', 
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable = disable_pbar):
        nwn.update_conductance()
        vs = get_node_voltages(
            nwn=nwn, signal=electrode_signals[t,:])
        edge_voltages = vs[nwn.edgelist[:,0]] - vs[nwn.edgelist[:,1]]
        nwn.update_flux(edge_voltages=edge_voltages, dt=dt)
        node_voltages[t,:] = vs
        if return_flux:
            edge_fluxes[t,:] = nwn.flux
        if update_signal and t+1<steps:
            electrode_signals = sig_augment(electrode_signals, node_voltages,nwn,t)

    if return_flux:
        return node_voltages, edge_fluxes  
    return node_voltages