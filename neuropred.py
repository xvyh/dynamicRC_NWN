#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# Neuromorphic reservoir computing - prediction task
Includes training and autonomous prediction code for 
multivariate time series 

Author: Yinhao Xu <yinhao.xu@sydney.edu.au>
"""
import numpy as np
import time 
import pickle
from neurowiresim import *

#|=============================[helper_functions]=============================|#
def pickle_dump(data, fname)-> None: 
    '''serialise data, make a .pkl file'''
    with open(fname, "wb") as output_file:
        pickle.dump(data, output_file)
    return None

def pickle_load(fname) -> None:
    '''read a .pkl file, deserialise data'''
    with open(fname, "rb") as input_file:
       return pickle.load(input_file)

def timed(func): #decorator for timing a function
    def inner(*args, **kwargs):
        init_time = time.time()
        start_time = time.strftime('\033[35m%Y/%m/%d;\033[95m %H:%M:%S', time.localtime())
        result = func(*args, **kwargs)
        pprint(f"|\033[36m{time.time()-init_time:->10.3f}s \033[32m[{func.__name__}]", end='')
        print(f" {start_time} -> {time.strftime('%H:%M:%S', time.localtime())}\033[0m|")
        return result
    return inner
    #comment out any `@timed` if you don't want to see function timing printing
#|============================================================================|#
#|----------------------------[coloured_printing]-----------------------------|#
current_clr = None #global colour variable

def clr(x=None, bg=False) -> str:
    """
        Return the ANSI colour code given some color_index. 
        Colour loops through RGYBM.
        x: (default None) Text colour. None for default (reset) ANSI.
        bg: (default False) Background colour
    """
    if x is None:
        return "\033[0m"
    elif isinstance(x, int): #looping bg colours RGYBM 
        return f"\033[{4 if bg else 3}{x%5+1}m"
    elif isinstance(x, str):
        if x =="*":  return "\033[1m" #bold
        elif x=="_": return "\033[4m" #underline
    else:
        raise ValueError 

def pprint(*args, **kwargs) -> None:
    """custom print() function. Adds a coloured | in the front."""
    global current_clr
    print(f'{clr(current_clr, bg=True)} {clr()}', end='')
    print(*args, **kwargs)
#|----------------------------------------------------------------------------|#
#|==================================[maths]===================================|#
@timed
def ridge_regression(Y, X, lmb:float=1e-6):
    """ridge regression; y = x W
    Y and X matrices with data stored in rows,
    Each row for different time step. 

    Args:
        Y (np matrix): expected output
        X (np matrix): training input
        lmb (float): lambda ridge parameter
    Returns:
        W (np matrix): matrix which minimises loss for Y = X @ W
    """
    XTX = X.T @ X
    XTY = X.T @ Y 
    I = np.identity(X.shape[1])
    return np.linalg.pinv(XTX + lmb*I) @ XTY

def MSE(Y, X): 
    """Mean Squared Error"""
    return np.mean((Y-X)**2)

#|============================================================================|#
#|-----------------------------------[data]-----------------------------------|#
def standardise(y,mu:float=None, sigma:float=None, return_mu_sigma:bool=False):
    """
    Standardisation (or Z-score normalisation)
    Normalise via mean and SD. Set mean->0, SD->1
    Args:
        y (np.ndarray): values to be normalised
    Returns array with 0 mean and 1 SD.
    """
    assert isinstance(y, np.ndarray), f"y is {type(y)}, not np.ndarray."
    if mu is None and sigma is None: 
        if len(y.shape)==2:
            mu = np.mean(y, axis=0) #mean
            sigma = np.std(y, axis=0) #SD
        elif len(y.shape)==1:
            mu = np.mean(y) 
            sigma = np.std(y)
        else:
            raise ValueError
    if return_mu_sigma:
        return (y-mu)/sigma, mu, sigma
    return (y-mu)/sigma

def gen_lorenz(T:int=5000, dt:float=1e-2, dt_data:bool=False, ics:list=None):
    """
    generate lorenz system values from the following
    initial condition (0, 1, 1.05)
    Default: T = 5000, dt = 0.01 
    T in time units of width dt, not seconds
    """
    def lorenz(xyz, s=10, r=28, b=2.6667):
        x,y,z = xyz
        xdot = s*(y-x)
        ydot = x*(r-z)-y
        zdot = x*y - b*z
        return np.array([xdot, ydot, zdot])
    xyzs = np.empty((T + 1, 3))  # Need one more for the initial values
    if ics is None:
        xyzs[0] = (17.67715816276679, 12.931379185960404, 43.91404334248268)  # Set initial values (0.,1.,1.05)#
    else:
        xyzs[0]=ics
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    dtdata = np.empty((T+1, 3))
    for i in range(T):
        dtdata[i] = lorenz(xyzs[i])
        xyzs[i + 1] = xyzs[i] + dtdata[i] * dt
    dtdata[T] = lorenz(xyzs[T])
    if dt_data:
        return xyzs, dtdata
    return xyzs 

def gen_input_output_electrodes(
        input_elec_setting, 
        out_elec_setting, 
        drain_electrodes, 
        n_input:int, n_total:int, n_readout:int,
        rng_seed, rng_seed2=None):
    input_electrodes_rng = np.random.default_rng(rng_seed)
    output_electrodes_rng = np.random.default_rng(rng_seed*2 if rng_seed2 is None else rng_seed2) #temporary arbitrary choice of rng seed
    #Dealing with input electrode input:
        # either 'random', or give a list, where n_inputs would be sampled from it
    if input_elec_setting is None:
        raise ValueError
    elif isinstance(input_elec_setting, np.ndarray):
        input_electrodes = input_elec_setting.tolist()
        pprint(f"|NOTICE (input electrodes): custom {len(input_electrodes)} electrodes.")
    elif input_elec_setting=="random":
        input_candidates = list(range(n_total))
        for i in drain_electrodes:
            input_candidates.remove(i)
        input_electrodes = input_electrodes_rng.choice(
            input_candidates, n_input, replace=False).tolist()
        pprint(f"|NOTICE (input electrodes): randomly selecting {n_input=} from {n_total=}.")
    elif isinstance(input_elec_setting, list):
        input_electrodes = input_elec_setting
        pprint(f"|NOTICE (input electrodes): custom {len(input_electrodes)} electrodes.")
    else:
        raise ValueError

    if len(input_electrodes) > n_input:
        input_candidates = input_electrodes
        for i in drain_electrodes:
            if i in input_candidates:
                input_candidates.remove(i)
        pprint(f"|NOTICE (input electrodes): randomly selecting {n_input=} from {len(input_candidates)} given nodes.")
        input_electrodes = input_electrodes_rng.choice(input_candidates, n_input, replace=False).tolist()

    #Dealing with output electrodes input:
    if out_elec_setting is None:
        raise ValueError
    elif out_elec_setting == "all":
        # Use all nodes as readout.
        assert n_total == n_readout 
        output_electrodes = list(range(n_total))
        pprint(f"|NOTICE (output electrodes): all {len(output_electrodes)} nodes as readouts.")
    elif out_elec_setting == "other":
        # Use all nodes other than input electrodes as readout.
        assert n_readout == n_total-n_input-len(drain_electrodes), f"{n_readout} readouts != {n_total-n_input-len(drain_electrodes)}"
        output_electrodes = list(range(n_total))
        #print(input_electrodes+drain_electrodes)
        for i in input_electrodes+drain_electrodes:
            output_electrodes.remove(i)
        pprint(f"|NOTICE (output electrodes): all none input/drain nodes => {len(output_electrodes)} readouts.")
    elif out_elec_setting == "other-":
        # Use some nodes other than input electrodes as readout.
        assert n_readout < n_total-n_input-len(drain_electrodes), f"{n_readout} readouts !< {n_total-n_input-len(drain_electrodes)}"
        output_electrodes_candidate = list(range(n_total))
        for i in input_electrodes+drain_electrodes:
            output_electrodes_candidate.remove(i)
        output_electrodes = output_electrodes_rng.choice(
            output_electrodes_candidate, n_readout, replace=False).tolist()
        pprint(f"|NOTICE (output electrodes): some none input/drain nodes => {len(output_electrodes)} readouts.")
    elif out_elec_setting == "other+":
        # Use all nodes other than input electrodes as readout, plus some input nodes.
        assert n_readout >= n_total-n_input-len(drain_electrodes)
        assert n_readout <= n_total-len(drain_electrodes)
        input_electrodes_not_readout = output_electrodes_rng.choice(
            input_electrodes, n_total-n_readout-len(drain_electrodes), replace=False).tolist()
        output_electrodes = list(range(n_total))
        for i in input_electrodes_not_readout+drain_electrodes:
            output_electrodes.remove(i)
        pprint(f"|NOTICE (output electrodes): none input/drain nodes + some input => {len(output_electrodes)} readouts.")
    elif isinstance(out_elec_setting, (list, np.ndarray)):
        output_electrodes=list(out_elec_setting)
        if len(output_electrodes) > n_readout:
            output_candidates = output_electrodes
            for i in drain_electrodes:
                if i in output_candidates:
                    output_candidates.remove(i)
            pprint(f"|NOTICE (output electrodes): randomly selecting {n_readout=} from {len(output_candidates)} given nodes.")
            output_electrodes = output_electrodes_rng.choice(output_candidates, n_readout, replace=False).tolist()
    else:
        raise ValueError 
    assert len(output_electrodes) == n_readout, f"{len(output_electrodes)=} not equal {n_readout=}"
    return input_electrodes, output_electrodes 

def square_lattice_network(W:int, H:int, torus:bool=False):
    """
    Generate the adjacency matrix of an undirected unweighted network, 
    which tiles with squares 

    Input:
        W (int): number of nodes for width of lattice grid
        H (int): number of nodes for height of lattice grid
        torus(bool): make the square lattice a torus?

    Returns:
        A (np.ndarray): undirected unweighted adjacency matrix
    
    E.g., for W=5, H=2, the returning network has form
        0-1-2-3-4
        | | | | |
        5-6-7-8-9
    
        if torus, then the connections circle back
        0-1-2-3-4 - 0
        etc...
    """
    tot = W*H
    A = np.zeros((tot, tot))
    for i in range(tot-1):
        last_row= i>= tot-W #if last row
        last_col= i%W==W-1 #if last col
        # if last_row and last_col only happen at (tot,tot)
        if last_row:
            A[i,i+1] = A[i+1,i] = 1
        elif last_col:
            A[i,i+W] = A[i+W,i] = 1
        else:
            A[i,i+1] = A[i+1,i] = 1
            A[i,i+W] = A[i+W,i] = 1

    if torus: #square into donut; connect the ends
        for i in range(0,tot,W): #connecting rows start and end
            A[i, i+W-1] = A[i+W-1, i] = 1
        for i in range(W): #connecting colns start and end
            A[i, i+W*(H-1)] = A[i+W*(H-1), i] = 1

    return A

def sqlattice_convert_to_id(coordinates:list, w:int):
    """ assume coordinates is a list of 2 element tuples [(x1,y1), (x2,y2),...] """
    return [int(x+w*y) for x,y in coordinates]
def sqlattice_convert_to_coord(ids:list, w:int):
    return [(id%w, id//w) for id in ids]
#|----------------------------------------------------------------------------|#
#|==========================[neuromorphic_machinery]==========================|#
@timed
def get_readout(electrodes,input_signal,return_flux,nwn):
    """
        get readout
        this function can be overwritten with any other RC framework
        for alternate implementations
    """
    return neuro_sim(
        nwn,electrodes, input_signal,
        dt = 0.05,steps=input_signal.shape[0],return_flux=return_flux)

@timed
def auto_pred(input_electrodes,readout_electrodes,input_signal,Win,bias,W,nwn,steps, y_init):
    """
        autonomous prediction 
        this function can be overwritten with any other RC framework
        for alternate implementations
    """
    saved_yhats = np.zeros((steps+1,3))
    saved_yhats[0,:] = y_init
    pred_buffer = 1
    def custom_electrode_signal_augment(
            electrode_signals, node_voltages, nwn, t, saved_yhats):
        saved_yhats[t+1] = node_voltages[t,readout_electrodes]@W + saved_yhats[t]
        if t>=pred_buffer:
            electrode_signals[t+1,:-1] = saved_yhats[t+1] @ Win + bias 
        return electrode_signals

    nodevs = neuro_sim(
        nwn,input_electrodes,
        input_signal,
        dt = 0.05,steps=steps,
        sig_augment=\
        lambda s,v,nwn,t: custom_electrode_signal_augment(
            s,v,nwn,t,saved_yhats))
    
    return nodevs[:,readout_electrodes], saved_yhats[1:,]

#|============================================================================|#
#|------------------------------[simulation_run]------------------------------|#
@timed
def run(
    input_signal,
    t_warmup:int,       #warmup timesteps
    t_train:int,        #training timesteps
    t_pred:int,         #prediction timesteps
    input_electrodes,
    output_electrodes,
    drain_electrodes,
    n_input:int,
    n_readout:int,
    n_total:int,
    alpha:float,
    lag:int=1,          #steps ahead prediction
    neuro_params:dict=None, 
    ridge_coeff:float=1e-6,
    exp_index:int=0,
    teacher_signal=None,
    return_flux:bool=False,
    dynamic_meas:bool=False,
    nonlin_meas:bool=False,
    ) -> dict:
    """
    Runs a neuromorphic prediction experiment.

    Parameters:
        input_signal (np.ndarray): The input signal for the experiment.
                        (int):  If int instead, it serves as the Lorenz signal
                                initialisation ID
        teacher_signal (np.ndarray): The teacher signal for the experiment.
                                    Usually same as input signal
        t_warmup (int): The number of warmup timesteps.
        t_train (int): The number of training timesteps.
        t_pred (int): The number of prediction timesteps.
        lag (int): The number of steps ahead for prediction.
        neuro_params (dict, optional): A dictionary of neuromorphic parameters. 
                Includes e.g. all the electrodes, nwn params, 
                number of nodes, Win, Win_bias. Defaults to None.
        ridge_coeff (float, optional): The coefficient for ridge regression. 
                Defaults to 1e-6.
        exp_index (int, optional): The index of the experiment. Defaults to 0.

    Returns:
        dict: A dictionary containing the results of the experiment, 
                including the readouts, the weight matrix W,
                and the predicted readouts if prediction duration is not zero.
    """

    pprint(clr(exp_index)+"-"*27
           +f"[{clr()}{clr('*')}{exp_index: ^4}{clr()}{clr(exp_index)}]"
           +"-"*27+clr()) #coloured dividing bar before simulation run

    #|____________________________(preprocessing)_____________________________|#
    if dynamic_meas:
        return_flux = True 
    # input signal
    if isinstance(input_signal, int):
        signal_ic_id = input_signal #Lorenz signal initialisation ID
        lorenz_dt = 0.005
        signal_for_ics = gen_lorenz(T=(signal_ic_id+1), dt=lorenz_dt)
        signal_data = gen_lorenz(T=t_pred+lag-1, dt=lorenz_dt, ics=signal_for_ics[signal_ic_id])
        # if original_tend is not None:
        original_signal_data = gen_lorenz(
            T=29000, dt=lorenz_dt, ics=signal_for_ics[signal_ic_id])
        _, org_mu, org_sigma = standardise(original_signal_data, return_mu_sigma=True)
        input_signal = standardise(signal_data, org_mu, org_sigma, False)
        num_signals = 3
    if teacher_signal is None:
        teacher_signal=input_signal
    if not isinstance(input_signal, np.ndarray):
        input_signal = np.array(input_signal)
    if not isinstance(teacher_signal, np.ndarray):
        teacher_signal = np.array(teacher_signal)
    if len(input_signal.shape)==1:
        input_signal = np.atleast_2d(input_signal).T 
    if len(teacher_signal.shape)==1:
        teacher_signal =  np.atleast_2d(teacher_signal).T
    save_data = {} #data to be serialised and saved
    train_dur = t_train - t_warmup 
    pred_dur  = t_pred  - t_train

    #electrodes
    input_electrodes, readout_electrodes=gen_input_output_electrodes(
        input_elec_setting=input_electrodes,
        out_elec_setting=output_electrodes,
        drain_electrodes=drain_electrodes,
        n_input=n_input,n_total=n_total, n_readout=n_readout,
        rng_seed=exp_index
    )
    #input weights
    Win_rng = np.random.default_rng(exp_index+10)
    Win = Win_rng.uniform(-alpha,alpha,size=(num_signals, n_input))
    #input bias
    Win_bias_rng = np.random.default_rng(exp_index+20)
    Win_bias = Win_bias_rng.uniform(alpha*2.5,alpha*5,size=(n_input))*(Win_bias_rng.integers(0,2,size=(n_input))*2-1)
    #reservoir input
    resin = input_signal @ Win + Win_bias
    #augment with drain node (=0 signal)
    drain_signal = np.atleast_2d(np.zeros(input_signal.shape[0])).T
    resin = np.concatenate((resin,drain_signal),axis=1)
    #|________________________(neuromorphic machinery)________________________|#
    # This section can be rewritten for alternate reservior implemetations
    nwn = NanowireNetwork(**neuro_params)
    returned_vals = get_readout(
        list(input_electrodes)+list(drain_electrodes),
        resin[:t_train],return_flux,nwn)
    if return_flux:
        output_signal,fluxes = returned_vals
    else:
        output_signal = returned_vals
    readout_signal = output_signal[:,readout_electrodes]
    save_data["readouts"] = readout_signal

    #|____________________________(postprocessing)____________________________|#
    Y=teacher_signal[t_warmup+lag:t_train+lag]-teacher_signal[t_warmup:t_train]
    X=readout_signal[t_warmup:t_train]
    W=ridge_regression(Y, X, lmb = ridge_coeff)# Training for output layer
    pprint(f"|{X.shape=}, {Y.shape=}, {W.shape=}.")
    yhat_train = X@W+teacher_signal[t_warmup:t_train]
    train_mse = MSE(Y, yhat_train)
    pprint(f"|Train MSE: [\033[34m{train_mse:.3e}\033[0m]")
    save_data["W"] = W

    #|________________________(autonomous prediction)_________________________|#
    # This section can be rewritten for alternate reservior implemetations
    pred_resin = np.zeros((pred_dur, n_input+1))
    pred_resin[0:2,:] = resin[t_train:t_train+2]
    if pred_dur !=0:
        pred_readout_signal, yhats = auto_pred(
            list(input_electrodes)+list(drain_electrodes),
            readout_electrodes,
            pred_resin, Win, Win_bias, W, nwn, pred_dur, 
            teacher_signal[t_train,:])
        if len(pred_readout_signal.shape)==1:
            pred_readout_signal =  np.atleast_2d(pred_readout_signal).T
        save_data["pred_readouts"] = pred_readout_signal 
        
        Yp = teacher_signal[t_train+lag:t_pred]
        Xp = pred_readout_signal[:pred_dur-lag]
        yhat_pred = Xp@W+yhats[:-lag]
        pred_mse = MSE(Yp, yhat_pred)
        save_data["yhat"] = yhat_pred
        pprint(f"|Prediction MSE: [\033[34m{pred_mse:.3e}\033[0m]")
    #|____________________________(dynamic measure)____________________________|#
    if dynamic_meas:
        save_tblock = int(1/lorenz_dt)*100 #100 Ly times
        save_ts_skip = int(1/lorenz_dt) #SAVE EVERY LY TIME 
        save_ts = list(range(max(t_train-save_tblock,0), t_train, save_ts_skip))
        pprint(f"{save_tblock=}, "
                f"[{t_train-save_tblock}, {t_train}]")
        divcount  = np.zeros((3,nwn.edgenum), dtype=np.int16)

        for j,k in enumerate(save_ts):
            A= np.abs(fluxes[k,:])/0.015
            Adiv = A.copy()
            Adiv[A>(2/3)] = 2
            Adiv[(A<0.55)] = -1
            Adiv[(A>=0.55)&(A<=(2/3))]=0
            if j==0:
                last_Adiv = Adiv
            Adivcount = Adiv - last_Adiv
            divcount[0]+=np.logical_or(Adivcount==1,Adivcount==-1).astype(np.int16)
            divcount[1]+=np.logical_or(Adivcount==2,Adivcount==-2).astype(np.int16)
            divcount[2]+=np.logical_or(Adivcount==3,Adivcount==-3).astype(np.int16)
            last_Adiv = Adiv
        save_data['divcounts'] = divcount

    #|____________________________(nonlin measure)____________________________|#
    if nonlin_meas: 
        nonlin_thres = min(11000, t_train) #only 10k steps to reach convergence
        neuro_params['eta']=0
        readout_lin_signal= get_readout(
            list(input_electrodes)+list(drain_electrodes),
            resin[:nonlin_thres],False,
            NanowireNetwork(**neuro_params))[:,readout_electrodes]
        nonlin_meas = np.mean(np.abs(
            readout_signal[1000:nonlin_thres] - \
            readout_lin_signal[1000:nonlin_thres])**2,axis=1)
        nonlin_meas_location = np.mean( #respecting node location
            np.abs(readout_signal[1000:nonlin_thres] - \
                   readout_lin_signal[1000:nonlin_thres])**2, axis=0)
        save_data["nonlin"] = nonlin_meas
        save_data["nonlin_loc"] = nonlin_meas_location
    pprint(clr(exp_index) + "_"*60 + clr())
    return save_data
#|----------------------------------------------------------------------------|#
