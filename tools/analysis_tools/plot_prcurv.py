import re, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fire


schema = 'thres,pr,rec'.split(',')


def main(fds='', save_plot_path=''):
    print(fds)
    print('--------------',save_plot_path)
    fds = fds.split(',')
    
    fns = filter(lambda x:(x.startswith('tpr') and not x.endswith('png')), os.listdir(fds[0]))
    color='rgbcmyk'
    for i,fn in enumerate(fns):
        plt.figure(i)
        plt.title(fn)
        for j,fd in enumerate(fds):
            df1 = pd.read_csv(fd+"/%s"%fn, header=None, names=schema)
            label_name =  fd.split('/')[-5] +'_' + fd.split('/')[-1].split('checkpoint_')[1].split('.')[0]
            plt.plot(df1['rec'], df1['pr'], "*-%s"%color[j], label=label_name.replace('prefusion_',''))
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(save_plot_path, '%s.png'%fn))
        
        
if __name__ == '__main__':
    fire.Fire(main)