import numpy as np 
from neuropred import * 

import matplotlib
import matplotlib.pyplot as plt 
from matplotlib import patheffects as pe

def fig_readouts_alpha(pkl_fname:str)->None:
    sdatas = pickle_load(pkl_fname)
    yhats = [
        sdatas[i]['yhat'] for i in [0,1,2]
    ]
    readouts = [sdatas[i]['readouts'] for i in [0,1,2]]
    Ws = [sdatas[i]['W'] for i in [0,1,2]]
    #compute forecast time
    lambda_max=0.91
    lorenz_data = gen_lorenz(52000,0.005)
    mu = np.mean(lorenz_data, axis=0) #mean
    sigma = np.std(lorenz_data, axis=0) #SD

    ic_id = 0
    t_cutoff2=27000
    post_att = 2000
    plot_cut =27000 + post_att
    pred_len = 2000
    lag=1
    #the individual signal evolutions
    true_sig = lorenz_data[t_cutoff2+lag+ic_id:plot_cut+ic_id,:]
    yhat_unnorms = [None]*3
    prediction_error = lambda y, yhat: np.sum((yhat-y)**2, axis=1)/np.mean(np.sum((y-y.mean())**2, axis=1))
    theta=0.4
    for i in [0,1,2]:
        pred_si = yhats[i]*sigma+mu
        total_frames = true_sig.shape[0]
        err_pred = prediction_error(lorenz_data[27000+1+ic_id:27000+pred_len +ic_id,:], pred_si[:pred_len -1, :])
        forecast_timestep = np.argmax(err_pred>theta)
        forecast_time = forecast_timestep*0.005*lambda_max
        yhat_unnorms[i] = pred_si
        print(f"{forecast_time=:.3f} Ly times")

    gs_kw = dict(width_ratios=[1,0.3], height_ratios=[1,1,1,1,1,1])
    fig, ax = plt.subplot_mosaic(
        [
        ['B0','A0'],
        ['C0','A0'],
        ['B1','A1'],
        ['C1','A1'],
        ['B2','A2'],
        ['C2','A2'],
        ],empty_sentinel="X",
        gridspec_kw=gs_kw, 
        figsize=(13, 8))
    a,b = 25000,27000
    xdata = np.linspace(a*0.005,b*0.005,b-a+1)[:-1]
    xlim= [125,135]
    fig.subplots_adjust(wspace=0.05, hspace=0.26)
    index_labels = ['(a) $\\alpha=$0.016 V','(b) $\\alpha=$0.2 V','(c) $\\alpha=$2.0 V']
    for i in [0,1,2]:
        outputs1 = readouts[i]
        W = Ws[i]
        num_of_ws_plot = 10
        plot_w = [None]*3
        for j in [0,1,2]:
            plot_w[j] = abs(W[:,j])
        wsum = np.array(plot_w).sum(axis=0)

        best_ws = np.argsort(wsum)[-num_of_ws_plot-1:-1]
        worst_ws = np.argsort(wsum)[0:num_of_ws_plot]

        ax[f'B{i}'].plot(xdata,outputs1[a:b, best_ws],c='r', linewidth=0.5, alpha=1, zorder=3)
        ax[f'C{i}'].plot(xdata,outputs1[a:b,worst_ws],c='m', linewidth=0.5, alpha=1, zorder=3)
        ylim0 =  1.05*np.array(ax[f'B{i}'].get_ylim())
        ylim1 =  1.05*np.array(ax[f'C{i}'].get_ylim())
        ax[f'B{i}'].plot(xdata,outputs1[a:b],c='0.75', linewidth=0.1, alpha=0.5, zorder=1)
        ax[f'C{i}'].plot(xdata,outputs1[a:b],c='0.75', linewidth=0.1, alpha=0.5, zorder=1)
        ax[f'B{i}'].set_ylim(ylim0)
        ax[f'C{i}'].set_ylim(ylim1)

        ax[f'B{i}'].tick_params(axis='x',which='both',bottom=False,labelbottom=False)
        ax[f'B{i}'].minorticks_on()
        ax[f'C{i}'].minorticks_on()
        ax[f'B{i}'].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: f'{x:>4.2f}'))
        ax[f'C{i}'].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: f'{x:>4.2f}'))
        ax[f'B{i}'].set_xlim(xlim)
        ax[f'C{i}'].set_xlim(xlim)
        ax[f'B{i}'].set_ylabel("$r_{{\\rm out}}$ (V)", fontsize=11)
        ax[f'C{i}'].set_ylabel("$r_{{\\rm out}}$ (V)", fontsize=11)
        ax[f'C{i}'].set_xlabel("$\\lambda_{{\\rm max}}t$", fontsize=11)
        pos = ax[f'C{i}'].get_position()

        ax[f'C{i}'].set_position([pos.x0, pos.y0 + 0.025, pos.width, pos.height])
        ax[f'C{i}'].xaxis.set_label_coords(0.5, -0.3)
        ax[f'B{i}'].text(0.012,0.70,index_labels[i],fontsize=15,transform=ax[f'B{i}'].transAxes,
                        path_effects=[pe.withStroke(linewidth=5, foreground="white")])

        ax[f'A{i}'].get_yaxis().set_visible(False)
        ax[f'A{i}'].get_xaxis().set_visible(False)
        ax[f'A{i}'].remove()
        ax[f'A{i}'] = fig.add_subplot(ax[f'A{i}'].get_subplotspec(), projection='3d')
        ax[f'A{i}'].xaxis.pane.fill = False
        ax[f'A{i}'].yaxis.pane.fill = False
        ax[f'A{i}'].zaxis.pane.fill = False
        ax[f'A{i}'].set_xticks([])
        ax[f'A{i}'].set_yticks([])
        ax[f'A{i}'].set_zticks([])
        ax[f'A{i}'].grid(False)
        ax[f'A{i}'].view_init(15, 120)
        ax[f'A{i}'].plot(yhat_unnorms[i][0:post_att,0], yhat_unnorms[i][0:post_att,1], yhat_unnorms[i][0:post_att,2], c='orange', lw=0.5, markersize=0.8, marker='.')
        ax[f'A{i}'].set_box_aspect(None, zoom=1.3)
        ax[f'A{i}'].set_facecolor('none')
        true_pre_att = 18000
        true_post_att = 3000
        ax[f'A{i}'].plot(
            lorenz_data[true_pre_att: true_pre_att+true_post_att,0], 
            lorenz_data[true_pre_att: true_pre_att+true_post_att,1], 
            lorenz_data[true_pre_att: true_pre_att+true_post_att,2], 
            lw=0.1, c='0.7')

    fig.savefig('readouts_w_scale', dpi=300, bbox_inches='tight')

def main() -> None:
    global current_clr
    current_clr = exp_index = 0
    with open("networks/500nw_9905jns.npy", 'rb') as f:
        adj_mtx = np.load(f)
    sdatas = [None]*3
    for i, alpha in enumerate([0.02,0.2,2]):
        sdatas[i] = run(
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
            alpha           = alpha,
            neuro_params    = {'adjmtx':adj_mtx},
            exp_index       = 343,
            return_flux     = False,
            dynamic_meas    = False,
            nonlin_meas     = False
            )
    pickle_dump(sdatas, f"saved_exp{exp_index:0>3}.pkl")
    fig_readouts_alpha(f"saved_exp{exp_index:0>3}.pkl")
if __name__ == "__main__": main()