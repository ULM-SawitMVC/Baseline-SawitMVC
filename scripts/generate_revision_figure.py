"""Data-exact implementation of the requested Image Gen design for Fig. 2.

Image Gen output: figures/paper/fig04_imagegen.png; prompt is saved alongside.
Generated bar geometry was approximate, so this plot renders source CSV values
with the same side-by-side design. No generated raster is edited.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def main():
    res, out = ROOT/'results/revision', ROOT/'figures/paper'
    ladder = pd.read_csv(res/'a2_error_ladder.csv').set_index('stage')
    data = pd.read_csv(res/'a10_verified_metrics.csv')
    gt = data[(data.condition=='gt') & (data.features=='F0') & (data.model=='ElasticNet')].macro.iloc[0]*100
    sweep = pd.read_csv(res/'a9_threshold_sweep_full.csv')
    sweep = sweep[(sweep.features=='F_all') & (sweep.model=='Ridge')].sort_values('threshold')
    labels = ['GT detections\n+ counter','Matched IDs,\nGT classes','Matched IDs,\nclass vote',
              'Class vote +\nunmatched boxes','Deployed\ncounter','Naive detection\nsum']
    stages = ['L1_recall','L2_recall_class','L3_full_detector','L4_deployed_counter','naive_sum_detector']
    values = [gt]+[ladder.loc[s,'macro']*100 for s in stages]
    blue, orange, red = '#2553a4','#bc5914','#b44b48'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
        'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65})
    fig = plt.figure(figsize=(7.16,3.15),facecolor='white')
    a = fig.add_axes([.155,.17,.315,.65])
    b = fig.add_axes([.615,.17,.37,.65])
    fig.text(.012,.96,'(a) Diagnostic counting rules',fontsize=9.3,va='top')
    fig.text(.565,.96,'(b) Threshold sensitivity',fontsize=9.3,va='top')
    bars = a.barh(np.arange(6),values,height=.61,
        color=[blue,'#e9f0fa','#fff1e1','#fff1e1',red,'#ecedef'],
        edgecolor=[blue,blue,orange,orange,red,'#555b63'],linewidth=.8)
    a.invert_yaxis()
    a.set_yticks(np.arange(6),labels)
    a.tick_params(axis='y',length=0,pad=5)
    a.set_xlim(0,114)
    a.set_xticks([0,25,50,75,100])
    a.set_xlabel('Class ±1 accuracy (%)',labelpad=7)
    a.grid(axis='x',color='#e5e5e5',linewidth=.5)
    a.set_axisbelow(True)
    for rect,val in zip(bars,values):
        a.text(val+1.7,rect.get_y()+rect.get_height()/2,f'{val:.2f}',va='center',fontsize=7.5)
    b.plot(sweep.threshold,sweep.macro*100,color=orange,linewidth=1.25,marker='o',
        markersize=3.6,markerfacecolor='white',markeredgewidth=1.0)
    b.axhline(gt,color=blue,linewidth=.9,linestyle=(0,(4,3)))
    b.axvline(.25,color='#929292',linewidth=.8,linestyle=(0,(1.5,2.5)))
    b.set_xlim(0,.75)
    b.set_ylim(60,102)
    b.set_xticks(np.arange(0,.71,.1))
    b.set_yticks([60,70,80,90,100])
    b.set_xlabel('Confidence threshold τ',labelpad=7)
    b.set_ylabel('Class ±1 accuracy (%)',labelpad=6)
    b.grid(color='#e5e5e5',linewidth=.5)
    b.set_axisbelow(True)
    legend = [Line2D([0],[0],color=blue,linestyle=(0,(4,3)),linewidth=.9,label=f'GT: {gt:.2f}'),
        Line2D([0],[0],color='#929292',linestyle=(0,(1.5,2.5)),linewidth=.8,label='Released τ = 0.25')]
    b.legend(handles=legend,loc='lower left',bbox_to_anchor=(-.01,1.045),ncol=2,
        frameon=False,fontsize=7,borderaxespad=0,columnspacing=1.1,handlelength=2.1)
    last = sweep.iloc[-1]
    b.annotate(f'{last.macro*100:.2f}',(last.threshold,last.macro*100),
        xytext=(-9,0),textcoords='offset points',ha='right',va='center',fontsize=7.4)
    fig.savefig(out/'fig04_attribution_sweep.pdf',facecolor='white')
    fig.savefig(out/'fig04_attribution_sweep.png',dpi=400,facecolor='white')
    plt.close(fig)
    pd.DataFrame({'label':[s.replace('\n',' ') for s in labels],'accuracy_pct':values}).to_csv(
        out/'fig04_panel_a_data.csv',index=False)
    print('Generated data-exact full-width figure from CSVs.')


if __name__=='__main__':
    main()
