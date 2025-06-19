import numpy as np 
from neuropred import * 

import matplotlib
import matplotlib.pyplot as plt 
from matplotlib import patheffects as pe

def fig_prediction(pkl_name:str)->None:
    sdata = pickle_load(pkl_name)
    lambda_max = 0.91
    theta = 0.4
    prediction_error = lambda y, yhat: np.sum((yhat-y)**2, axis=1)/np.mean(np.sum((y-y.mean())**2, axis=1))
    yhat = sdata['yhat']
    lorenz_data = gen_lorenz(52000,0.005)
    mu = np.mean(lorenz_data, axis=0) #mean
    sigma = np.std(lorenz_data, axis=0) #SD
    lorenz_norm = standardise(lorenz_data)
    yhat_unnorm = yhat*sigma+mu
    ic_id = sdata['ic_id']
    t_cutoff2=27000
    pred_len = 2000
    plot_cut = t_cutoff2 +pred_len
    true_sig = lorenz_norm[t_cutoff2+1+ic_id:plot_cut+ic_id,:]
    pred_sig = yhat
    total_frames = true_sig.shape[0]
    lytime = np.linspace(0, total_frames*0.005, total_frames)*lambda_max
    err_pred = prediction_error(lorenz_data[27000+1+ic_id:27000+pred_len +ic_id,:], yhat_unnorm[:pred_len -1, :])
    forecast_timestep = np.argmax(err_pred>theta)
    forecast_time = forecast_timestep*0.005*lambda_max

    gs_kw = dict(width_ratios=[1, 0.5], height_ratios=[1,1,1],hspace=0,wspace=0)
    fig, ax = plt.subplot_mosaic([[0,'a'],[1,'a'],[2,'a']],gridspec_kw=gs_kw, figsize=(12, 4))
    line_clr = 'orangered'
    line_style = '--'
    true_line_clr = '0.5'
    true_line_width = pred_line_width = 1
    for i in [0,1,2]:
        ax[i].set(ylabel=f'$\\hat{{y}}_{i}$', xlim=[0,lytime[-1]])
        ax[i].plot(lytime, true_sig[:, i], color=true_line_clr, linewidth=true_line_width)
        ax[i].plot(lytime[:], pred_sig[:plot_cut-t_cutoff2-1, i],c=line_clr, linestyle=line_style, lw=pred_line_width)[0]
        ax[i].minorticks_on()
        ax[i].axvline(x=forecast_time, c='0.3', linestyle='--', linewidth=1)
        ax[i].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: f'{x:>3.0f}'))

    ax[0].text(0.01,0.95, "(a)", horizontalalignment='left', verticalalignment='top', size=16, transform = ax[0].transAxes)
    ax[1].text(0.01,0.95, "(b)", horizontalalignment='left', verticalalignment='top', size=16, transform = ax[1].transAxes)
    ax[2].text(0.01,0.95, "(c)", horizontalalignment='left', verticalalignment='top', size=16, transform = ax[2].transAxes)

    ax[2].set(xlabel='$\lambda_{{\\rm max}} t$')
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks(np.arange(min(lytime), max(lytime), 1.0))
    ax['a'].remove()
    ax['a'] = fig.add_subplot(ax[f'a'].get_subplotspec(), projection='3d')
    pre_att = t_cutoff2 # first attractor plot index for training part
    post_att = int(50//(0.005*lambda_max))
    true_pre_att = 27000
    true_post_att = 2000
    ax['a'].plot(
        lorenz_data[true_pre_att:pre_att+true_post_att,0], 
        lorenz_data[true_pre_att:pre_att+true_post_att,1], 
        lorenz_data[true_pre_att:pre_att+true_post_att,2], 
        lw=1, c='0.3', label='true')
    ax['a'].plot(
        yhat_unnorm[0:post_att,0], 
        yhat_unnorm[0:post_att,1], 
        yhat_unnorm[0:post_att,2], c='orangered', lw=1, label='predicted')
    ax['a'].xaxis.pane.fill = False
    ax['a'].yaxis.pane.fill = False
    ax['a'].zaxis.pane.fill = False
    ax['a'].set_xticks([])
    ax['a'].set_yticks([])
    ax['a'].set_zticks([])
    ax['a'].grid(False)
    ax['a'].view_init(15, 120)
    ax['a'].set_box_aspect(None, zoom=1.3)
    ax['a'].legend()
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig('pred.png', dpi=300, bbox_inches='tight')

def main() -> None:
    global current_clr
    current_clr = exp_index = 0
    with open("networks/500nw_9905jns.npy", 'rb') as f:
        adj_mtx = np.load(f)
    sdata = run(
        input_signal    = 0,
        t_warmup        = 100,
        t_train         = 27000,
        t_pred          = 29000,
        input_electrodes= 'random',
        output_electrodes='other',
        drain_electrodes= [336],
        n_input         = 24,
        n_readout       = 475,
        n_total         = 500,
        alpha           = 0.2,
        neuro_params    = {'adjmtx':adj_mtx},
        exp_index       = 0,
        return_flux     = False,
        dynamic_meas    = False,
        nonlin_meas     = False
        )
    pickle_dump(sdata, f"saved_exp{exp_index:0>3}.pkl")
    fig_prediction(f"saved_exp{exp_index:0>3}.pkl")
if __name__ == "__main__": main()