#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:46:19 2017

@author: kwilmes
"""
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rcParams
from scipy import stats
import matplotlib.cm as cmaps
from analyse_experiment import *
#import colormaps as cmaps


interval = .54
#for timeframe in [nonplasticwarmup,warmup+.54,warmup+rewardsimtime+.24,total]:
#for 70ms 20 ms gap
interval = .63
halfinterval = .28
input_time = .07



def plot_alltuningcurves(tunings_initial,tuning_rewardend,tuning_after_rewardend,tunings_final, N_neurons,N_pop,save=None,name = ''):
        ymax = 4    
        if N_pop > 1:
            fig = plt.figure(figsize=(3.1,2.6)) # 4.2, 3
            if name == '':
                ymax = 4
        else:
            fig = plt.figure(figsize=(1.7,1.5))
            plt.xlabel('orientation')
            plt.ylabel('firing rate')
        groupsize = N_neurons/N_pop
        for k in range(0,N_neurons,int(groupsize)):    
            if N_pop > 1:
                plt.subplot(N_pop/2,N_pop/2,(k/groupsize)+1)
            if not tunings_initial is None:
                plt.plot(tunings_initial[k:int(k+groupsize),:].T,'k',lw=.5, label = 'before')
            if not tuning_rewardend is None:
                plt.plot(tuning_rewardend[k:int(k+groupsize),:].T,color = cmaps.magma(0.2),lw=.5, label='before end of rewarded phase')
            if not tuning_after_rewardend is None:
                plt.plot(tuning_after_rewardend[k:int(k+groupsize),:].T,color = cmaps.viridis(0.5),lw=.5,label = 'after rewarded phase')
            if not tunings_final is None:
                plt.plot(tunings_final[k:int(k+groupsize),:].T, color = cmaps.magma(0.7),lw=.5, label = 'after refinement phase')
            plt.xticks(np.arange(0,N_pop),('|','/','--','\\'))
            plt.yticks(np.arange(0,ymax+1,2))
            
            if (k/groupsize)+1 == 3:
                plt.xlabel('orientation')
                plt.ylabel('firing rate')           

            plt.ylim(0,ymax)
            plt.tight_layout()
        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
            
        if save is not None:
            plt.savefig('%s/singletuning%s.eps'%(save,name), bbox_extra_artists=(lgd,), bbox_inches='tight')




def plot_responsechangehistogram(tunings_initial,tunings_final, N_neurons,N_pop,ax,save=None,name = ''):
    groupsize = N_neurons/N_pop
    ind = np.arange(N_pop)  # the x locations for the groups
    width = 0.35       # the width of the bars
    pvalues = []
    initial_means = []
    initial_std = []
    final_means = []
    final_std = []

    for k in range(0,N_neurons,int(groupsize)):    
        ttest = stats.ttest_ind(tunings_initial[k:int(k+groupsize),0],tunings_final[k:int(k+groupsize),0])
        pvalues.append(ttest[1])    
        initial_means.append(np.mean(tunings_initial[k:int(k+groupsize),0],0))
        initial_std.append(np.std(tunings_initial[k:int(k+groupsize),0],0))
        final_means.append(np.mean(tunings_final[k:int(k+groupsize),0],0))
        final_std.append(np.std(tunings_final[k:int(k+groupsize),0],0))
     
    if ax is None:
        fig, ax = plt.subplots()
    rects1 = ax.bar(ind, initial_means, width, color='k', yerr=initial_std)    
    rects2 = ax.bar(ind + width, final_means, width, color = cmaps.magma(0.7), yerr=final_std)
    
    ax.set_ylabel('firing rate')
    ax.set_title('Response towards rewarded stimulus')
    
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('|', '/', '--', '\ '))
    
    ax.legend((rects1[0], rects2[0]), ('Before', 'After'))

    i = 0
    for rect in rects2:
        height = rect.get_height()
        ax.text(rect.get_x(), 1.05*height,
                '%.6f' % pvalues[i],
                ha='center', va='bottom')
        i+=1


    if save is not None:
        plt.tight_layout()
        plt.savefig('%s/barplot%s.pdf'%(save,name), bbox_extra_artists=(lgd,), bbox_inches='tight') 

def plot_SOMswitch(tunings_initial,tunings_final, N_neurons,N_pop,save=None,name = ''):
    groupsize = N_neurons/N_pop
    ind = np.arange(2)  # the x locations for the groups
    width = 0.35       # the width of the bars
    pvalues = []
    means = []
    std = []

    ttest = stats.ttest_ind(tunings_initial,tunings_final)
    pvalues.append(ttest[1])    
    print(ttest)
    means.append(np.mean(tunings_initial))
    std.append(np.std(tunings_initial))
    means.append(np.mean(tunings_final))
    std.append(np.std(tunings_final))
            
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, means, width, color='k', yerr=std)    

    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('firing rate')

    
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('with SST', 'without SST'))

    plt.tight_layout()
    plt.savefig('%s/SOMswitch%s.pdf'%(save,name), bbox_extra_artists=(lgd,), bbox_inches='tight') 


def plot_tuningcurves(tunings_initial,tuning_rewardend,tuning_after_rewardend,tunings_final, N_neurons,N_pop,save=None,name = ''):
        ymax = 4    
        if N_pop > 1:
            fig = plt.figure(figsize=(3.1,2.6)) # 4.2, 3
            if name == '':
                ymax = 4
        else:
            fig = plt.figure(figsize=(1.7,1.5))
            plt.xlabel('orientation')
            plt.ylabel('firing rate')
        groupsize = N_neurons/N_pop
        for k in range(0,N_neurons,int(groupsize)):    
            if N_pop > 1:
                plt.subplot(N_pop/2,N_pop/2,(k/groupsize)+1)
        

            plt.plot(np.arange(np.shape(tunings_initial)[1]),np.mean(tunings_initial[k:int(k+groupsize),:],0),'k',lw=3, label = 'before')
          
            if not tuning_rewardend is None:
                plt.plot(np.arange(np.shape(tunings_initial)[1]),np.mean(tuning_rewardend[k:int(k+groupsize),:],0),color = cmaps.magma(0.2),lw=3, label='before end of rewarded phase')
            if not tuning_after_rewardend is None:
                plt.plot(np.arange(np.shape(tunings_initial)[1]),np.mean(tuning_after_rewardend[k:int(k+groupsize),:],0),color = cmaps.viridis(0.5),lw=3,label = 'after rewarded phase')
            if not tunings_final is None:
                plt.plot(np.arange(np.shape(tunings_initial)[1]),np.mean(tunings_final[k:int(k+groupsize),:],0), color = cmaps.magma(0.7),lw=3, label = 'after refinement phase')
            plt.xticks(np.arange(0,np.shape(tunings_initial)[1]),('|','/','--','\\'))
            plt.yticks(np.arange(0,ymax+1,2))
            
            if (k/groupsize)+1 == 3:
                plt.xlabel('orientation')
                plt.ylabel('firing rate')
                
          
            plt.tight_layout()
        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
            

        if save is not None:
            plt.savefig('%s/tuning%s.eps'%(save,name), bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_grandavgtuningcurves(tunings_initial,tunings_final,N_pop,save,name = ''):
        ymax = 4    
        if N_pop > 1:
            fig = plt.figure(figsize=(7,5))
            if name == '':
                ymax = 3
        else:
            fig = plt.figure(figsize=(4,3))
            plt.xlabel('orientation')
            plt.ylabel('firing rate')
        for k in range(0,N_pop):    
            if N_pop > 1:
                plt.subplot(N_pop/2,N_pop/2,k+1)
                plt.title('%s population %d'%(name,k+1))
            else:
                plt.title('%s population'%(name))

            for m in range(np.shape(tunings_initial)[0]):
                plt.plot(np.arange(np.shape(tunings_initial)[2]),tunings_initial[m,k:k+1,:].T,'.7',lw=.5)
            plt.plot(np.arange(np.shape(tunings_initial)[2]),np.mean(tunings_initial[:,k:k+1,:],0).T,'k',lw=2)

            for m in range(np.shape(tunings_initial)[0]):
                plt.plot(np.arange(np.shape(tunings_initial)[2]),tunings_final[m,k:k+1,:].T, color = cmaps.viridis(0.55),lw=.5)
            plt.plot(np.arange(np.shape(tunings_initial)[2]),np.mean(tunings_final[:,k:k+1,:],0).T, color = cmaps.viridis(0.6),lw=2)


            plt.yticks(np.arange(0,np.shape(tunings_initial)[2]+1,1))
            
            if k+1 == 3:

                plt.xlabel('orientation')
                plt.ylabel('firing rate')
                

            plt.ylim(0,ymax)
            if name == '':
                plt.suptitle('excitatory tuning curves')
            else:
                plt.suptitle('%s tuning curves'%name)
            plt.tight_layout()
        lgd = plt.legend(['before','end'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
            

        plt.savefig('%s/tuninggrandavg%s.eps'%(save,name), bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_firingratechange(tunings_initial,tunings_final, save,name = ''):
        max_rate = np.max(tunings_final)
        fig = plt.figure(figsize=(3,3))
        
        plt.xlabel('firing rate before')
        plt.ylabel('firing rate after')
        plt.scatter(tunings_initial[:,0],tunings_final[:,0], color = cmaps.viridis(0.0),lw=1)
        plt.scatter(tunings_initial[:,1],tunings_final[:,1], color = cmaps.viridis(0.33),lw=1)
        plt.scatter(tunings_initial[:,2],tunings_final[:,2], color = cmaps.viridis(0.66),lw=1)
        plt.scatter(tunings_initial[:,3],tunings_final[:,3], color = cmaps.viridis(1.0),lw=1)
        plt.plot(np.arange(-.1,max_rate+.2,.1),np.arange(-.1,max_rate+.2,.1), color = 'k')
        plt.xlim(-.1,max_rate+.1)
        plt.ylim(-.1,max_rate+.1)
        plt.xticks(np.arange(0,max_rate+.1,1))
        plt.yticks(np.arange(0,max_rate+.1,1))
        
        plt.title('%s'%name)

        plt.tight_layout()
       
        plt.savefig('%s/firingratechange%s.eps'%(save,name))#, bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_currentratiochange(current_initial,current_final, save,name = ''):
        max_current = np.max(current_final)
        min_current = np.min(current_initial)

        fig = plt.figure(figsize=(3,3))
        
        plt.xlabel('E/(E+I) before')
        plt.ylabel('E/(E+I) after')
        plt.scatter(current_initial[:,0],current_final[:,0], color = cmaps.viridis(0.0),lw=1)
        plt.scatter(current_initial[:,1],current_final[:,1], color = cmaps.viridis(0.33),lw=1)
        plt.scatter(current_initial[:,2],current_final[:,2], color = cmaps.viridis(0.66),lw=1)
        plt.scatter(current_initial[:,3],current_final[:,3], color = cmaps.viridis(1.0),lw=1)
        plt.plot(np.arange(1,max_current+.2,.1),np.arange(1,max_current+.2,.1), color = 'k')
        plt.xlim(1,max_current+.1)
        plt.ylim(1,max_current+.1)

        plt.xticks(np.arange(1,max_current+.1,1))
        plt.yticks(np.arange(1,max_current+.1,1))
       

        plt.tight_layout()
        
        plt.savefig('%s/currentratiochange%s.eps'%(save,name))#, bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_currentchange(current_initial,current_final, save,name = ''):
        
        current_initial *= 10**9
        current_final *= 10**9 # to get values in nS
            
        max_current = np.max(current_final)
        min_current = np.min(current_final)
        fig = plt.figure(figsize=(3.2,3))
        
        plt.xlabel('%s before [nA]'%name)
        plt.ylabel('%s after [nA]'%name)
        plt.scatter(current_initial[:,0],current_final[:,0], color = cmaps.viridis(0.0),lw=1,label = '|')
        plt.scatter(current_initial[:,1],current_final[:,1], color = cmaps.viridis(0.33),lw=1,label = '/')
        plt.scatter(current_initial[:,2],current_final[:,2], color = cmaps.viridis(0.66),lw=1,label = '\\')
        plt.scatter(current_initial[:,3],current_final[:,3], color = cmaps.viridis(1.0),lw=1,label = '--')
        if min_current > 0:
            min_current = 0
        plt.plot(np.arange(min_current-.1,max_current+.2,.1),np.arange(min_current-.1,max_current+.2,.1), color = 'k')
        if name == 'E':
            plt.xlim(min_current,max_current+.1)
            plt.ylim(min_current,max_current+.1)
            plt.xticks(np.arange(np.round(min_current,1),max_current+.1,.5))
            plt.yticks(np.arange(np.round(min_current,1),max_current+.1,.5))
        else:
            plt.xlim(min_current,max_current+.05)
            plt.ylim(min_current,max_current+.05)
            plt.xticks(np.arange(np.round(min_current,1),max_current+.05,.1))
            plt.yticks(np.arange(np.round(min_current,1),max_current+.05,.1))
        

        plt.tight_layout()
        
        plt.savefig('%s/currentchange%s.eps'%(save,name))#, bbox_extra_artists=(lgd,), bbox_inches='tight')#, bbox_extra_artists=(lgd,), bbox_inches='tight')



def rasterplots(neuront, neuroni, timeframe,start,end, name):
    plt.figure(figsize=(5,2))
    a1 = plt.subplot(111)
    plt.grid(True, color='k', linestyle='-', linewidth=.5)

    if name == 'PYR':
        plot(neuront[neuroni<100]/second, neuroni[neuroni<100], '.', color = cmaps.viridis(0.0), ms=2.0)
        plot(neuront[((neuroni>=100)&(neuroni<200))]/second, neuroni[((neuroni>=100)&(neuroni<200))], '.', color = cmaps.viridis(0.33), ms=2.0)
        plot(neuront[((neuroni>=200)&(neuroni<300))]/second, neuroni[((neuroni>=200)&(neuroni<300))], '.', color = cmaps.viridis(0.66), ms=2.0)
        plot(neuront[neuroni>=300]/second, neuroni[neuroni>=300], '.', color = cmaps.viridis(1.0), ms=2.0)
        plt.yticks(np.arange(0,np.max(neuroni)+100,100))
    elif name == 'SST':
        plot(neuront[neuroni<30]/second, neuroni[neuroni<30], '.', color = cmaps.viridis(0.0), ms=2.0)
        plot(neuront[((neuroni>=30)&(neuroni<60))]/second, neuroni[((neuroni>=30)&(neuroni<60))], '.', color = cmaps.viridis(0.33), ms=2.0)
        plot(neuront[((neuroni>=60)&(neuroni<90))]/second, neuroni[((neuroni>=60)&(neuroni<90))], '.', color = cmaps.viridis(0.66), ms=2.0)
        plot(neuront[neuroni>=90]/second, neuroni[neuroni>=90], '.', color = cmaps.viridis(1.0), ms=2.0)
        plt.yticks(np.arange(0,np.max(neuroni)+30,30))
    elif name == 'PV':
        plot(neuront/second, neuroni, '.k', ms=2.0)
        plt.yticks(np.arange(0,np.max(neuroni)+10,120))
    else:
        plot(neuront/second, neuroni, '.k', ms=2.0)
    
    xlabel('time [seconds]')
    ylabel('neuron index')
    xticks(np.arange(timeframe-interval,timeframe+input_time,input_time))


    xlim(timeframe-interval,timeframe)
    a1.spines['bottom'].set_visible(False)
    a1.spines['left'].set_visible(False)
    a1.spines['top'].set_visible(False)
    a1.spines['right'].set_visible(False)
    a1.tick_params(axis='both', which='both', length=0)
    a1.axvspan(start,end, facecolor='.8',lw=0.0, alpha=0.5)

    plt.savefig('%s/%s/%s/%srasterplot%d.pdf'%(savepath,dataname,run_no,name,timeframe),rasterized=True)#, bbox_extra_artists=(lgd,), bbox_inches='tight') 


def firing_rates(neuront, neuroni, time, rates, name):    
    plt.figure()        
    a1 = plt.subplot(211)
    plt.grid(True, color='w', linestyle='-', linewidth=2)
    plt.gca().patch.set_facecolor('0.8')
    plt.plot(neuront/second, neuroni, 'k.',ms=2.0)
    xlabel('time [seconds]')
    ylabel('neuron index')
    yticks([])
    plt.xticks(np.arange(timeframe-interval,timeframe,input_time))
    a1.spines['bottom'].set_visible(False)
    a1.spines['left'].set_visible(False)
    a1.spines['top'].set_visible(False)
    a1.spines['right'].set_visible(False)
    a1.tick_params(axis='both', which='both', length=0)

    if name == 'PYR':
        plt.yticks(np.arange(0,np.max(neuroni),100))
    elif name == 'SST':
        plt.yticks(np.arange(0,np.max(neuroni),30))
    else:
        pass
    plt.xlim(timeframe-interval,timeframe)
    plt.subplot(212)
    plt.ylabel('population firing rate [Hz]')
    if len(rates) == 4:
        plt.plot(time/second, rates[0]/Hz, color = cmaps.viridis(0.0))
        plt.plot(time/second, rates[1]/Hz, color = cmaps.viridis(0.33))
        plt.plot(time/second, rates[2]/Hz, color = cmaps.viridis(0.66))
        plt.plot(time/second, rates[3]/Hz, color = cmaps.viridis(1.0))
        lgd=plt.legend(['%s1'%name,'%s2'%name,'%s3'%name,'%s4'%name],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)

    else:
        plt.plot(time/second, rates[0]/Hz, color = '.4')
        lgd=plt.legend(['%s'%name],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
        
    plt.ylim(0,300)
    plt.xlim(timeframe-interval,timeframe)
    plt.xticks(np.arange(timeframe-interval,timeframe,input_time))
    plt.xlabel('time [seconds]')
    plt.savefig('%s/%s/%s/%sfiring_rates%d.eps'%(savepath, dataname, run_no,name,timeframe), bbox_extra_artists=(lgd,), bbox_inches='tight',rasterized=True)
    

def currentplots(time, exc_currents, inh_currents, timefrom, timeto, name):
        plt.figure(figsize=(7,5))
        for i in range(len(exc_currents)):
            plt.subplot(2,2,i+1)
            plot(time/second, exc_currents[i]/nA,'k')
            plot(time/second, inh_currents[i]/nA, '.75')
            plt.xlabel('time [ms]')
            plt.ylabel('current')
            plt.xlim(timefrom,timeto+.1)
            plt.ylim(-1,1)
            title('%s%d currents'%(name,i+1))
        plt.tight_layout()
        lgd=plt.legend(['Exc','Inh'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
        plt.savefig('%s/%s/%s/%scurrents%d.eps'%(savepath,dataname,run_no,name,timeto), bbox_extra_artists=(lgd,), bbox_inches='tight') 



def tsplot(ax, data,**kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    ax.plot(x,est,**kw,lw=2)
    ax.margins(x=0)
    
    
def plot_xcorr(pyr1,pyr2,sst,pv, condition=None):
    fig = plt.figure(figsize=(3.5,5.5))        
    ax2 = fig.add_subplot(311)
    ax2.xcorr(pyr1, pv, usevlines=True, normed=False, maxlags=maxlags, lw=5, color = '.6')
    
    plt.yticks(np.arange(0,70,30),np.arange(0,70,30)) 
    plt.xticks(np.arange(-maxlags,maxlags+5,5),np.arange(-maxlags*binsize,maxlags*binsize+5,5*binsize)) 
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')

    plt.title('PC1 - PV')
    
    ax3 = fig.add_subplot(312)
    ax3.xcorr(sst, pv, usevlines=True, normed=False, maxlags=maxlags, lw=5, color = '.6')
   
    plt.xticks(np.arange(-maxlags,maxlags+5,5),np.arange(-maxlags*binsize,maxlags*binsize+5,5*binsize)) 
    plt.yticks(np.arange(0,35,15),np.arange(0,35,15)) 
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')

    plt.title('SST1 - PV')

    ax5 = fig.add_subplot(313)
    ax5.xcorr(pyr1, pyr2, usevlines=True, normed=False, maxlags=maxlags, lw=5, color = '.6')
 
    plt.xticks(np.arange(-maxlags,maxlags+5,5),np.arange(-maxlags*binsize,maxlags*binsize+5,5*binsize)) 
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.yaxis.set_ticks_position('left')
    ax5.xaxis.set_ticks_position('bottom')

    plt.yticks(np.arange(0,15,5),np.arange(0,11,5)) 
    plt.title('PC1 - PC2')
    plt.xlabel('time lag [ms]')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig('%s/%s/%s/xcorr_%s.eps'%(savepath, dataname, run_no, condition))

def plot_xcorr2(pyr1,pyr2,sst,pv, condition=None):
    fig = plt.figure(figsize=(3.5,5.5))        
    ax2 = fig.add_subplot(311)
    ax2.xcorr(pyr1, sst, usevlines=True, normed=False, maxlags=maxlags, lw=5, color = '.6')
  
    plt.yticks(np.arange(0,70,30),np.arange(0,70,30)) 
    plt.xticks(np.arange(-maxlags,maxlags+5,5),np.arange(-maxlags*binsize,maxlags*binsize+5,5*binsize)) 
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')

    plt.title('PC1 - SST')
    
    ax3 = fig.add_subplot(312)
    ax3.xcorr(pyr2, sst, usevlines=True, normed=False, maxlags=maxlags, lw=5, color = '.6')
   
    plt.xticks(np.arange(-maxlags,maxlags+5,5),np.arange(-maxlags*binsize,maxlags*binsize+5,5*binsize)) 
    plt.yticks(np.arange(0,35,15),np.arange(0,35,15)) 
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')

    plt.title('PYR2 - SST')

    ax5 = fig.add_subplot(313)
    ax5.xcorr(pyr2, pv, usevlines=True, normed=False, maxlags=maxlags, lw=5, color = '.6')
   
    plt.xticks(np.arange(-maxlags,maxlags+5,5),np.arange(-maxlags*binsize,maxlags*binsize+5,5*binsize)) 
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.yaxis.set_ticks_position('left')
    ax5.xaxis.set_ticks_position('bottom')

    plt.yticks(np.arange(0,15,5),np.arange(0,11,5)) 
    plt.title('PC2 - PV')
    plt.xlabel('time lag [ms]')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig('%s/%s/%s/xcorr_%s.eps'%(savepath, dataname, run_no, condition))


if __name__ == "__main__":
    dataname = 'Spiking_model'
    savepath = './'
    Wrecafterreward = False
    Wrecafterwarmup = False
    xcorr = False # plot correllograms if True
    plastic = False # plot all synaptic weight changes if True
    reader = ExperimentReader('./%s'%dataname)
    # load all runs at once
    runs = reader.get_all_experiment_runs()
    run_nos = np.arange(1,2).astype(str) # TD removed, BCM type PYR-VIP 
    # look at the config for run 1
    no_stimuli = runs[run_nos[0]]['config']['params']['N4']
    N_pyr = runs[run_nos[0]]['config']['params']['NPYR']
    N_pop = 4
    N_sst = runs[run_nos[0]]['config']['params']['NSOM']
    N_pv = runs[run_nos[0]]['config']['params']['NPV']
    N_vip = runs[run_nos[0]]['config']['params']['NVIP']
    seed = runs[run_nos[0]]['config']['params']['seed']
    nonplasticwarmup = runs[run_nos[0]]['config']['params']['nonplasticwarmup_simtime']['py/reduce'][1]['py/tuple'][0]['values']
    plasticwarmup = runs[run_nos[0]]['config']['params']['warmup_simtime']['py/reduce'][1]['py/tuple'][0]['values']
    rewardsimtime = runs[run_nos[0]]['config']['params']['reward_simtime']['py/reduce'][1]['py/tuple'][0]['values']
    norewardsimtime = runs[run_nos[0]]['config']['params']['noreward_simtime']['py/reduce'][1]['py/tuple'][0]['values']
    noSSTPVsimtime = runs[run_nos[0]]['config']['params']['noSSTPV_simtime']['py/reduce'][1]['py/tuple'][0]['values']

    gmax = runs[run_nos[0]]['config']['params']['gmax']['py/reduce'][1]['py/tuple'][0]['values']*siemens

    input_time = runs[run_nos[0]]['config']['params']['input_time']['py/reduce'][1]['py/tuple'][0]['values']
    warmup = nonplasticwarmup + plasticwarmup
    total = warmup+rewardsimtime+norewardsimtime+nonplasticwarmup+noSSTPVsimtime

    t = np.arange(.0,135.3,.0001)*second
    


    
    dep_param = np.zeros(len(run_nos)) 
    dep_param2 = np.zeros(len(run_nos)) 
    performance = np.zeros(len(run_nos))
    performance_binary = np.zeros(len(run_nos))
    W_sst_pv = np.zeros((len(run_nos),N_sst*N_pv))
    W_pyr = np.zeros((len(run_nos),N_pyr*N_pyr))
    W_pyr_i = np.zeros((len(run_nos),N_pyr*N_pyr))
    W_pyr_j = np.zeros((len(run_nos),N_pyr*N_pyr))
    con_SOM_VIP_i = np.zeros((len(run_nos),N_sst*N_vip))
    con_SOM_VIP_j = np.zeros((len(run_nos),N_sst*N_vip))
    con_VIP_SOM_i = np.zeros((len(run_nos),N_vip*N_sst))
    con_VIP_SOM_j = np.zeros((len(run_nos),N_vip*N_sst))

    r_pyr = np.zeros((len(run_nos),N_pyr))

    rel_select_increase_mean = np.zeros((len(run_nos)))
    rel_select_increase = np.zeros((len(run_nos),N_pop))
    resp_increase = np.zeros((len(run_nos),N_pop))
    rel_resp_increase = np.zeros((len(run_nos),N_pop))
    response_rel_to_max = np.zeros((len(run_nos),N_pop))
    response_rel_to_max_before = np.zeros((len(run_nos),N_pop))
    response = np.zeros((len(run_nos),N_pop))
    increase = np.zeros((len(run_nos),N_pop))
    simple_resp_increase = np.zeros((len(run_nos),N_pop))
    SST0_PVmean = np.zeros((len(run_nos)))
    SSTother_PVmean = np.zeros((len(run_nos)))
    impact = np.zeros((len(run_nos)))
    impact_afterreward = np.zeros((len(run_nos)))
    impactmax = np.zeros((len(run_nos)))
    impactmax_afterreward = np.zeros((len(run_nos)))
    sst_pv = np.zeros((len(run_nos)))
    tuning = np.zeros((len(run_nos)))
    SOM0VIPprob = np.zeros((len(run_nos),int(N_sst/30)))
    VIPSOM0prob = np.zeros((len(run_nos),int(N_sst/30)))

    tuning_initial = np.zeros((len(run_nos),N_pop, N_pop))
    tuning_final = np.zeros((len(run_nos),N_pop, N_pop))
    SSTtuning_initial = np.zeros((len(run_nos),N_pop, N_pop))
    SSTtuning_final = np.zeros((len(run_nos),N_pop, N_pop))
    PVtuning_initial = np.zeros((len(run_nos),N_pop, N_pop))
    PVtuning_final = np.zeros((len(run_nos),N_pop, N_pop))
    VIPtuning_initial = np.zeros((len(run_nos),N_pop, N_pop))
    VIPtuning_final = np.zeros((len(run_nos),N_pop, N_pop))


    W_sst_pv_means = np.zeros((len(run_nos),N_pop))
    W_sst_pv_std = np.zeros((len(run_nos),N_pop))
    W_sst_pv_means_afterreward = np.zeros((N_pop))
    W_sst_pv_std_afterreward = np.zeros((N_pop))

    varied_param2 = 'tau_spikelet'
    varied_param = 'p_PV_PYR'

    
    plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    plt.set_cmap(cmaps.viridis)
    
    checker = None
    
    for i, run_no in enumerate(run_nos):
        all_data = reader.try_loading_artifacts(run_no)
        
        # get parameter
        config = runs[run_no]['config']

        dep_param[i] = runs[run_no]['config']['params'][varied_param]#['py/reduce'][1]['py/tuple'][0]['values']
        dep_param2[i] = runs[run_no]['config']['params'][varied_param2]['py/reduce'][1]['py/tuple'][0]['values']
        

        results = all_data['results.pkl']
  

        PYRi = results['PYRi'][:]
        PYRt = results['PYRt'][:]  
        SSTi = results['SSTi'][:]
        SSTt = results['SSTt'][:]     
        PVi = results['PVi'][:]
        PVt = results['PVt'][:]     
        VIPi = results['VIPi'][:]
        VIPt = results['VIPt'][:]     

      
        # recurrent weights

        W_rec = results['weights_rec']
        
        # second population
        try:
            W_rec2 = results['weights_rec2']
            checker = True
            print('two excitatory populations')

        except KeyError:
            print('one excitatory population')
            
            
        W_pyr_i = results['weights_rec_i'][:]
        W_pyr_j = results['weights_rec_j'][:]

        SOMPV_t = results['SOMPV_t'][:]
        SOMPV_w = results['SOMPV_w'][:]
        SOMVIP_w = results['SOMVIP_w'][:]
        SOMSOM_w = results['SOMSOM_w'][:]
        VIPSOM_w = results['VIPSOM_w'][:]
        PYRPV_w = results['PYRPV_w'][:]
        PYRVIP_w = results['PYRVIP_w'][:]
        PVPYR_w = results['PVPYR_w'][:]
        try:
            PYRSOM1_w = results['PYRSOM1_w'][:]
            PYRSOM2_w = results['PYRSOM2_w'][:]
            PYRSOM3_w = results['PYRSOM3_w'][:]
            PYRSOM4_w = results['PYRSOM4_w'][:]
        except:
            print('excepted PYRSOM')
        SOMPYR_w = results['SOMPYR_w'][:]
        PVVIP_w = results['PVVIP_w'][:]
        PVSOM_w = results['PVSOM_w'][:]
        PVPV_w = results['PVPV_w'][:]
        VIPVIP_w = results['VIPVIP_w'][:]
        VIPPYR_w = results['VIPPYR_w'][:]
        VIPPV_w = results['VIPPV_w'][:]

        # tuning
        try:
            tunings_before = results['tuning_initial']
            tunings_after = results['tuning_final']
            tunings_initial = results['tuning_initial']
            tunings_final = results['tuning_final']
            tunings_afterwarmup = results['tuning_afterwarmup']
            tuning_rewardend = results['tuning_rewardend']
            tuning_after_rewardend = results['tuning_after_rewardend']
            SOMtunings_initial = results['SOMtuning_initial']
            SOMtunings_final = results['SOMtuning_final']
            SOMtuning_rewardend = results['SOMtuning_rewardend']
            SOMtuning_after_rewardend = results['SOMtuning_after_rewardend']
            PVtunings_initial = results['PVtuning_initial']
            PVtunings_final = results['PVtuning_final']
            PVtuning_rewardend = results['PVtuning_rewardend']
            PVtuning_after_rewardend = results['PVtuning_after_rewardend']
            PVtuning_afterwarmup = results['PVtuning_afterwarmup']
            VIPtunings_initial = results['VIPtuning_initial']
            VIPtunings_final = results['VIPtuning_final']
            VIPtuning_rewardend = results['VIPtuning_rewardend']
            VIPtuning_after_rewardend = results['VIPtuning_after_rewardend']
            VIPtuning_afterwarmup = results['VIPtuning_afterwarmup']    
            stimtunings_initial = results['stimtuning_initial']
            stimtunings_final = results['stimtuning_final']
            stimPVtunings_initial = results['stimPVtuning_initial']
            stimPVtunings_final = results['stimPVtuning_final']
        except:
            print('minimal plotting')
            
        try:

            tuning_finalSOMswitch = results['tuning_finalSOMswitch']
            SOMswitch = True
        except:
            print('noSOMswitch')
            SOMswitch = False
        try:
            currentratio_initial=results['currentratio_initial']
            currentratio_final=results['currentratio_final']     
            ampE_initial=results['ampE_initial']
            ampE_final=results['ampE_final']    
            ampI_initial=results['ampI_initial']
            ampI_final=results['ampI_final'] 

        except:
            print('no current plotting')
            
        try:
            W_rec_afterreward = results['weights_rec_afterreward']
            Wrecafterreward = True
            W_rec_afterwarmup = results['weights_rec_afterwarmup']
            W_rec_start = results['weights_rec_start']
            Wrecafterwarmup = True
            W_sst_pv_afterreward = results['weights_sst_pv_afterreward'][:]

        except:
            print('no W_recs')


        try:
            impact[i] = results['impact']
            impact_afterreward[i] = results['impact_aftereward']
            impactmax[i] = results['impactmax']
            impactmax_afterreward[i] = results['impactmax_aftereward']
        except KeyError:
            print('no impact')
        
        try:
            PYR0toothers_2 = results['PYR0toothers_2'][:]
            otherstoPYR0_2 = results['otherstoPYR0_2'][:]
            PYR1toothers_2 = results['PYR1toothers_2'][:]
            PYR2toothers_2 = results['PYR2toothers_2'][:]
            tunings_initial2 = results['tuning_initial2']
            tunings_final2 = results['tuning_final2']
            tunings_afterwarmup2 = results['tuning_afterwarmup2']
            tuning_rewardend2 = results['tuning_rewardend2']
            tuning_after_rewardend2 = results['tuning_after_rewardend2']

        except KeyError:
            print('no second population')
        
        PYR0toothers = results['PYR0toothers'][:]
        otherstoPYR0 = results['otherstoPYR0'][:]
        PYR1toothers = results['PYR1toothers'][:]
        
        try:
            PYR1toPYR1 = results['PYR1toPYR1'][:]
            PYR2toPYR2 = results['PYR2toPYR2'][:]
            selfconnections = True
        except KeyError:
            selfconnections = False
        try:
            PYR3toothers = results['PYR3toothers'][:]
            otherstoPYR3 = results['otherstoPYR3'][:]
            PYR2toothers = results['PYR4toothers'][:]
            impact[i] = results['impact']
            impact_afterreward[i] = results['impact_afterreward']
            PYR3 = True
        except KeyError:
            PYR3 = False
            PYR2toothers = results['PYR2toothers'][:]
            print('no PYR3totohers')
        
        #population rates
        Pyrrate = []
        Pyrrate.append(results['Pyr1rate'])
        Pyrrate.append(results['Pyr2rate'])
        Pyrrate.append(results['Pyr3rate'])
        Pyrrate.append(results['Pyr4rate'])
        PVrate = []
        PVrate.append(results['PVrate'])
        SOMrate = []
        SOMrate.append(results['SOM1rate'])
        SOMrate.append(results['SOM2rate'])
        SOMrate.append(results['SOM3rate'])
        SOMrate.append(results['SOM4rate'])

        try: 
            SOM0toPV = results['SOM0PV'][:]
            SOMotherstoPV = results['SOMotherPV'][:]
        except KeyError:
            print('SOM0PV not in results.pkl')



        #look at connections
        
        # sst-to-pv connections

        sst_pv[i] = SST0_PVmean[i]-SSTother_PVmean[i]
        


        ind = np.arange(N_pop)  # the x locations for the groups
        width = 0.35       # the width of the bars
        groupsize = (N_sst*N_pv)/N_pop

        for k in range(N_pop):
            W_sst_pv_means[i,k] = np.mean(W_sst_pv[i,int(k*groupsize):int((k+1)*groupsize)]) 
            W_sst_pv_std[i,k] = np.std(W_sst_pv[i,int(k*groupsize):int((k+1)*groupsize)]) 
            W_sst_pv_means_afterreward[k] = np.mean(W_sst_pv_afterreward[int(k*groupsize):int((k+1)*groupsize)]) 
            W_sst_pv_std_afterreward[k] = np.std(W_sst_pv_afterreward[int(k*groupsize):int((k+1)*groupsize)]) 
        PYR1and2toothers = np.concatenate((PYR1toothers,PYR2toothers))

        
        if selfconnections == True:

            fig, (a1) = plt.subplots(1,1,figsize=(2,3))
            a1.axvspan((nonplasticwarmup+plasticwarmup)*10,(nonplasticwarmup+plasticwarmup+rewardsimtime)*10, facecolor='.8',lw=0.0, alpha=0.5)
            
            tsplot(a1,data=PYR1and2toothers[:,:]/nS,color = cmaps.viridis(1.0))#,label='others to others')
            tsplot(a1,data=PYR0toothers[:,:]/nS, color = cmaps.viridis(0.0))#,label ='1 to others')
            tsplot(a1,data=otherstoPYR0[:,:]/nS, color = cmaps.viridis(0.5))#,label='others to 1')
            tsplot(a1,data=PYR1toPYR1[:,:]/nS,color = cmaps.viridis(0.2))#,label='others to others')
            tsplot(a1,data=PYR2toPYR2[:,:]/nS,color = cmaps.viridis(0.7))#,label='others to others')

            a1.set_xlabel('time [seconds]')
            a1.set_ylabel('excitatory weights [nS]')
            a1.set_xticks(np.arange(0,total*10,200))
            a1.set_xticklabels(np.arange(0,int(total),20))
            lgd = a1.legend(['others to others','1 to others', 'others to 1','1 to 1','2 to 2'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
            plt.savefig('%s/%s/%s/weightchange.pdf'%(savepath,dataname,run_no), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

        """Other weights"""

        """Big figure"""
        fig, ((axa1,axa3),(axa2,axa4),(ax6, ax5)) = plt.subplots(3,2,figsize=(6.3,8), gridspec_kw = {'width_ratios':[1.2, 1]})
        axa1.axvspan((nonplasticwarmup+plasticwarmup)*10,(nonplasticwarmup+plasticwarmup+rewardsimtime)*10, facecolor='.8',lw=0.0, alpha=0.5)
        
        tsplot(axa1,data=PYR1and2toothers[:,:]/nS,color = cmaps.viridis(1.0))#,label='others to others')
        tsplot(axa1,data=PYR0toothers[:,:]/nS, color = cmaps.viridis(0.0))#,label ='1 to others')
        tsplot(axa1,data=otherstoPYR0[:,:]/nS, color = cmaps.viridis(0.5))#,label='others to 1')

        axa1.set_xlabel('time [seconds]')
        axa1.set_ylabel('excitatory weights [nS]')
        axa1.set_xticks(np.arange(0,total*10,200))
        axa1.set_xticklabels(np.arange(0,int(total),20))

        axa2.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
        tsplot(axa2,data=SOMPV_w[:3000,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
        tsplot(axa2,data=SOMPV_w[3100:6100,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
        tsplot(axa2,data=SOMPV_w[6200:9200,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
        tsplot(axa2,data=SOMPV_w[9300:12300,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')

        axa2.set_xlabel('time [seconds]')
        axa2.set_ylabel('SST-to-PV weights [nS]')
        axa2.set_ylim(-.02,1.02)
        axa2.set_xticks(np.arange(0,total,20))
        axa2.set_xticklabels(np.arange(0,int(total),20))



        plot_synapses(W_pyr_i, W_pyr_j, W_rec/gmax, var_name='normalised exciatory weights',
                           plot_type='scatter', cmap='viridis', rasterized=True, axes =  axa3)
        add_background_pattern(axa3)
        axa3.set_xlabel('presynaptic')
        axa3.set_xticks(np.arange(0,N_pyr+10,100))

        axa3.set_ylabel('postsynaptic')
        axa3.set_yticks(np.arange(0,N_pyr+10,100))
        axa3.set_title('impact=%.2f'%(impact[i]))


        
        
        ind = np.arange(N_pop)+.5 # the x locations for the groups
        width = 0.35       # the width of the bars
        groupsize = (N_sst*N_pv)/N_pop
        a = cmaps.viridis(0.0)
        

        axa4.errorbar(ind, W_sst_pv_means[i,:]/nS, yerr=W_sst_pv_std[i,:]/nS, color='.75', ls = ' ', marker='o', capsize=5, capthick=1, ecolor='black')


        x0,x1 = axa3.get_xlim()
        y0,y1 = axa3.get_ylim()
        axa4.set_aspect( 4.0/1.0 )

        axa4.set_ylabel('final SST-to-PV weights [nS]')
        axa4.set_xlabel('SST population')
        axa4.set_xticks(ind)
        axa4.set_xticklabels(('|','/','--','\\'))
        axa4.set_ylim(-.02,1.02)
        axa4.set_xlim(0,N_pop)

        if checker == True:
            image = plot_synapses(W_pyr_i, W_pyr_j, W_rec2/gmax, var_name='normalised excitatory weights',
                                  plot_type='scatter', cmap='viridis', rasterized=True, axes=ax5)
        else: 
            image = plot_synapses(W_pyr_i, W_pyr_j, W_rec/gmax, var_name='normalised excitatory weights',
                                  plot_type='scatter', cmap='viridis', rasterized=True, axes=ax5)



        add_background_pattern(ax5)
        ax5.set_xlabel('presynaptic')
        ax5.set_xticks(np.arange(0,N_pyr+10,100))

        ax5.set_ylabel('postsynaptic')
        ax5.set_yticks(np.arange(0,N_pyr+10,100))
        ax5.set_title('|%s=%.2f|%s=%.2f'%(varied_param,dep_param[i]*10**10,varied_param2,dep_param2[i]*10**10))

            
        ax6.axvspan((nonplasticwarmup+plasticwarmup)*10,(nonplasticwarmup+plasticwarmup+rewardsimtime)*10, facecolor='.8',lw=0.0, alpha=0.5)
        if checker == True:
            PYR1and2toothers_2 = np.concatenate((PYR1toothers_2,PYR2toothers_2))

            tsplot(ax6,data=PYR1and2toothers_2[:,:]/nS,color = cmaps.viridis(1.0))#,label='others to others')
            tsplot(ax6,data=PYR0toothers_2[:,:]/nS, color = cmaps.viridis(0.0))#,label ='1 to others')
            tsplot(ax6,data=otherstoPYR0_2[:,:]/nS, color = cmaps.viridis(0.5))#,label='others to 1')
        else: 
            tsplot(ax6,data=PYR1and2toothers[:,:]/nS,color = cmaps.viridis(1.0))#,label='others to others')
            tsplot(ax6,data=PYR0toothers[:,:]/nS, color = cmaps.viridis(0.0))#,label ='1 to others')
            tsplot(ax6,data=otherstoPYR0[:,:]/nS, color = cmaps.viridis(0.5))#,label='others to 1')


        ax6.set_xlabel('time [seconds]')
        ax6.set_ylabel('excitatory weight [nS]')
        ax6.set_xticks(np.arange(0,total*10,200))
        ax6.set_xticklabels(np.arange(0,int(total),20))
        lgd = plt.legend(['others to others','1 to others','others to 1'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
        
        plt.tight_layout()

        
        plt.savefig('%s/%s/%s/2ndbigfigure.pdf'%(savepath,dataname,run_no), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

        for timeframe in [nonplasticwarmup,warmup+nonplasticwarmup+interval,warmup+rewardsimtime+halfinterval,warmup+rewardsimtime+interval,60.06,warmup+rewardsimtime-interval-interval,90.86,114.8,total+rewardsimtime-nonplasticwarmup]:

            
            f, (a1, a2, a3) = plt.subplots(3, sharex=True, figsize = (5,6))
            a1.plot(PYRt[PYRi<100]/second, PYRi[PYRi<100], '.', color = cmaps.viridis(0.0), ms=2.0)
            a1.plot(PYRt[((PYRi>=100)&(PYRi<200))]/second, PYRi[((PYRi>=100)&(PYRi<200))], '.', color = cmaps.viridis(0.33), ms=2.0)
            a1.plot(PYRt[((PYRi>=200)&(PYRi<300))]/second, PYRi[((PYRi>=200)&(PYRi<300))], '.', color = cmaps.viridis(0.66), ms=2.0)
            a1.plot(PYRt[PYRi>=300]/second, PYRi[PYRi>=300], '.', color = cmaps.viridis(1.0), ms=2.0)
            a1.grid(True, color='k', linestyle='-', linewidth=.5)
            a1.set_title('PC')

            a1.set_xlim(timeframe-interval,timeframe)

            a1.set_yticks(np.arange(0,np.max(PYRi)+100,100))
            
            a2.plot(PVt/second, PVi, '.k', ms=2.0)
            a2.set_yticks(np.arange(0,np.max(PVi)+10,120))
            a2.grid(True, color='k', linestyle='-', linewidth=.5)
            a2.set_title('PV')

            a2.set_xlim(timeframe-interval,timeframe)


            a3.plot(SSTt[SSTi<30]/second, SSTi[SSTi<30], '.', color = cmaps.viridis(0.0), ms=2.0)
            a3.plot(SSTt[((SSTi>=30)&(SSTi<60))]/second, SSTi[((SSTi>=30)&(SSTi<60))], '.', color = cmaps.viridis(0.33), ms=2.0)
            a3.plot(SSTt[((SSTi>=60)&(SSTi<90))]/second, SSTi[((SSTi>=60)&(SSTi<90))], '.', color = cmaps.viridis(0.66), ms=2.0)
            a3.plot(SSTt[SSTi>=90]/second, SSTi[SSTi>=90], '.', color = cmaps.viridis(1.0), ms=2.0)
            a3.grid(True, color='k', linestyle='-', linewidth=.5)

            a3.set_xlim(timeframe-interval,timeframe)

            a3.set_yticks(np.arange(0,np.max(SSTi)+30,30))
            if timeframe == nonplasticwarmup:
                xlabel('time [seconds]')
                ylabel('neuron index')

            a3.set_xticks(np.arange(timeframe-interval,timeframe+input_time-.01,input_time))
            a3.set_title('SST')

            for a in [a1, a2, a3]:
                a.spines['bottom'].set_visible(False)
                a.spines['left'].set_visible(False)
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)
                a.tick_params(axis='both', which='both', length=0)
                a.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)

            plt.savefig('%s/%s/%s/bigrasterplot%s.pdf'%(savepath,dataname,run_no,timeframe),rasterized=True)
            
        if plastic == True:
            fig, ((a1,a3),(a2,a4),(a6, a5)) = plt.subplots(3,2,figsize=(6.3,8), gridspec_kw = {'width_ratios':[1.2, 1]})
            a1.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
            tsplot(a1,data=PVPYR_w[:12000,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
            tsplot(a1,data=PVPYR_w[12000:24000,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
            tsplot(a1,data=PVPYR_w[24000:36000,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
            tsplot(a1,data=PVPYR_w[36000:48000,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')
    
            a1.set_xlabel('time [seconds]')
            a1.set_ylabel('PV-to-PYR weights [nS]')
            a1.set_ylim(-.02,1.02)

            a2.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
            tsplot(a2,data=PYRPV_w[:8463,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
            tsplot(a2,data=PYRPV_w[11284:19747,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
            tsplot(a2,data=PYRPV_w[22568:31031,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
            tsplot(a2,data=PYRPV_w[33852:42315,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')
    
            a2.set_xlabel('time [seconds]')
            a2.set_ylabel('PYR-to-PV weights [nS]')
            a2.set_ylim(-.02,1.02)

            a3.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
            tsplot(a3,data=VIPSOM_w[:1500,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
            tsplot(a3,data=VIPSOM_w[1500:3000,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
            tsplot(a3,data=VIPSOM_w[3000:4500,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
            tsplot(a3,data=VIPSOM_w[4500:6000,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')
    
            a3.set_xlabel('time [seconds]')
            a3.set_ylabel('VIP-to-SOM weights [nS]')
            a3.set_ylim(-.02,1.02)

            a4.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
            tsplot(a4,data=SOMVIP_w[:1500,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
            tsplot(a4,data=SOMVIP_w[1500:3000,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
            tsplot(a4,data=SOMVIP_w[3000:4500,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
            tsplot(a4,data=SOMVIP_w[4500:6000,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')
    
            a4.set_xlabel('time [seconds]')
            a4.set_ylabel('SOM-to-VIP weights [nS]')
            a4.set_ylim(-.02,1.02)

            a5.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
            tsplot(a5,data=PYRVIP_w[:5000,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
            tsplot(a5,data=PYRVIP_w[5000:10000,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
            tsplot(a5,data=PYRVIP_w[10000:15000,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
            tsplot(a5,data=PYRVIP_w[15000:20000,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')
    
            a5.set_xlabel('time [seconds]')
            a5.set_ylabel('PYR-to-VIP weights [nS]')
            a5.set_ylim(-.02,1.02)

            a6.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
            tsplot(a6,data=PYRSOM1_w[:,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
            tsplot(a6,data=PYRSOM2_w[:,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
            tsplot(a6,data=PYRSOM3_w[:,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
            tsplot(a6,data=PYRSOM4_w[:,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')
    
            a6.set_xlabel('time [seconds]')
            a6.set_ylabel('PYR-to-SOM weights [nS]')
            a6.set_ylim(-.02,1.02)

            plt.tight_layout()
            plt.savefig('%s/%s/%s/otherweights.pdf'%(savepath,dataname,run_no), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 
    
    
            fig, ((a1,a3),(a2,a4),(a6, a5),(a7,a8)) = plt.subplots(4,2,figsize=(6.3,10), gridspec_kw = {'width_ratios':[1.2, 1]})
            a1.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
            tsplot(a1,data=SOMPYR_w[:12000,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
            tsplot(a1,data=SOMPYR_w[12000:24000,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
            tsplot(a1,data=SOMPYR_w[24000:36000,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
            tsplot(a1,data=SOMPYR_w[36000:48000,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')
    
            a1.set_xlabel('time [seconds]')
            a1.set_ylabel('SOM-to-PYR weights [nS]')
            a1.set_ylim(-.02,1.02)

            a2.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
            tsplot(a2,data=PVVIP_w[:1500,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
            tsplot(a2,data=PVVIP_w[1500:3000,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
            tsplot(a2,data=PVVIP_w[3000:4500,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
            tsplot(a2,data=PVVIP_w[4500:6000,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')
    
            a2.set_xlabel('time [seconds]')
            a2.set_ylabel('PV-to-VIP weights [nS]')
            a2.set_ylim(-.02,1.02)
    
            a3.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
            tsplot(a3,data=PVPV_w[:3600,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
            tsplot(a3,data=PVPV_w[3600:7200,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
            tsplot(a3,data=PVPV_w[7200:10800,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
            tsplot(a3,data=PVPV_w[10800:14400,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')
    
            a3.set_xlabel('time [seconds]')
            a3.set_ylabel('PV-to-PV weights [nS]')
            a3.set_ylim(-.02,1.02)

    
            a4.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
            tsplot(a4,data=SOMSOM_w[:3600,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
            tsplot(a4,data=SOMSOM_w[3600:7200,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
            tsplot(a4,data=SOMSOM_w[7200:10800,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
            tsplot(a4,data=SOMSOM_w[10800:14400,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')
    
            a4.set_xlabel('time [seconds]')
            a4.set_ylabel('SOM-to-SOM weights [nS]')
            a4.set_ylim(-.02,1.02)

            a5.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
            tsplot(a5,data=VIPPYR_w[:5000,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
            tsplot(a5,data=VIPPYR_w[5000:10000,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
            tsplot(a5,data=VIPPYR_w[10000:15000,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
            tsplot(a5,data=VIPPYR_w[15000:20000,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')
    
            a5.set_xlabel('time [seconds]')
            a5.set_ylabel('VIP-to-PYR weights [nS]')
            a5.set_ylim(-.02,1.02)

            a6.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
            tsplot(a6,data=PVSOM_w[:3600,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
            tsplot(a6,data=PVSOM_w[3600:7200,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
            tsplot(a6,data=PVSOM_w[7200:10800,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
            tsplot(a6,data=PVSOM_w[10800:14400,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')
    
            a6.set_xlabel('time [seconds]')
            a6.set_ylabel('PV-to-SOM weights [nS]')
            a6.set_ylim(-.02,1.02)

            
            a7.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
            tsplot(a7,data=VIPPV_w[:1500,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
            tsplot(a7,data=VIPPV_w[1500:3000,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
            tsplot(a7,data=VIPPV_w[3000:4500,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
            tsplot(a7,data=VIPPV_w[4500:6000,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')
    
            a7.set_xlabel('time [seconds]')
            a7.set_ylabel('VIP-to-PV weights [nS]')
            a7.set_ylim(-.02,1.02)
    
            a8.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
            tsplot(a8,data=VIPVIP_w[:,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')

            a8.set_xlabel('time [seconds]')
            a8.set_ylabel('VIP-to-VIP weights [nS]')
            a8.set_ylim(-.02,1.02)
            
            
            plt.tight_layout()
            plt.savefig('%s/%s/%s/otherweights2.pdf'%(savepath,dataname,run_no), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 


        
       
        plot_tuningcurves(tunings_initial,tuning_rewardend,tuning_after_rewardend,tunings_final,N_pyr,N_pop,save='%s/%s/%s'%(savepath, dataname, run_no),name = '')
        plot_tuningcurves(SOMtunings_initial,SOMtuning_rewardend,SOMtuning_after_rewardend,SOMtunings_final, N_sst,N_pop,save='%s/%s/%s'%(savepath, dataname, run_no),name = 'SST')
        plot_tuningcurves(PVtunings_initial,PVtuning_rewardend,PVtuning_after_rewardend,PVtunings_final, N_pv,1,save='%s/%s/%s'%(savepath, dataname, run_no),name = 'PV')
        
        plot_responsechangehistogram(tunings_initial,tunings_final, N_pyr, N_pop, None, save='%s/%s/%s'%(savepath, dataname, run_no), name = '')
        plot_responsechangehistogram(tunings_initial,tunings_afterwarmup, N_pyr, N_pop, None, save='%s/%s/%s'%(savepath, dataname, run_no), name = 'afterwarmup')
       
        
        fig, ((axa1,axa5),(axa2,axa3),(axa6, axa4)) = plt.subplots(3,2,figsize=(6.3,8), gridspec_kw = {'width_ratios':[1.2, 1]})
        axa1.axvspan((nonplasticwarmup+plasticwarmup)*10,(nonplasticwarmup+plasticwarmup+rewardsimtime)*10, facecolor='.8',lw=0.0, alpha=0.5)
        tsplot(axa1,data=PYR1and2toothers[:,:]/nS,color = cmaps.viridis(1.0))#,label='others to others')
        tsplot(axa1,data=PYR0toothers[:,:]/nS, color = cmaps.viridis(0.0))#,label ='1 to others')
        tsplot(axa1,data=otherstoPYR0[:,:]/nS, color = cmaps.viridis(0.5))#,label='others to 1')

        axa1.set_xlabel('time [seconds]')
        axa1.set_ylabel('excitatory weights [nS]')
        axa1.set_xticks(np.arange(0,(nonplasticwarmup+plasticwarmup+rewardsimtime)*10,100))
        axa1.set_xticklabels(np.arange(0,int((nonplasticwarmup+plasticwarmup+rewardsimtime)),10))
        axa1.set_xlim(0,(nonplasticwarmup+plasticwarmup+rewardsimtime)*10)

        axa2.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
        tsplot(axa2,data=SOMPV_w[:3600,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
        tsplot(axa2,data=SOMPV_w[3600:7200,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
        tsplot(axa2,data=SOMPV_w[7200:10800,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
        tsplot(axa2,data=SOMPV_w[10800:14400,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')

        axa2.set_xlabel('time [seconds]')
        axa2.set_ylabel('SST-to-PV weights [nS]')
        axa2.set_ylim(-.02,1.02)
        axa2.set_xlim(0,(nonplasticwarmup+plasticwarmup+rewardsimtime))
        axa2.set_xticks(np.arange(0,(nonplasticwarmup+plasticwarmup+rewardsimtime),10))
        axa2.set_xticklabels(np.arange(0,int((nonplasticwarmup+plasticwarmup+rewardsimtime)),10))

        
        plot_synapses(W_pyr_i, W_pyr_j, W_rec_start*siemens/gmax, var_name='normalised excitatory weights',
                           plot_type='scatter', cmap='viridis', rasterized=True, axes =  axa5, vmax =.5)
        add_background_pattern(axa5)
        axa5.set_xlabel('presynaptic')
        axa5.set_xticks(np.arange(0,N_pyr+10,100))
        #axa5.set_xticklabels(('1','2','3','4'))
        axa5.set_ylabel('postsynaptic')
        axa5.set_yticks(np.arange(0,N_pyr+10,100))
        #axa5.set_yticklabels('10','2','3','4'))
       
        
        

        plot_synapses(W_pyr_i, W_pyr_j, W_rec_afterwarmup*siemens/gmax, var_name='normalised excitatory weights',
                           plot_type='scatter', cmap='viridis', rasterized=True, axes =  axa3, vmax =.5)
        #add_background_pattern(axa3)
        axa3.set_xlabel('')
        axa3.set_xticklabels([])
        axa3.set_ylabel('')
        axa3.set_yticklabels([])

        plot_synapses(W_pyr_i, W_pyr_j, W_rec_afterreward*siemens/gmax, var_name='normalised excitatory weights',
                           plot_type='scatter', cmap='viridis', rasterized=True, axes =  axa4, vmax=.5)
        #add_background_pattern(axa4)
        axa4.set_ylabel('')
        axa4.set_yticklabels([])
        axa4.set_xlabel('')
        axa4.set_xticklabels([])

        

        ind = np.arange(N_pop)+.5 # the x locations for the groups
        width = 0.35       # the width of the bars
        groupsize = (N_sst*N_pv)/N_pop
        a = cmaps.viridis(0.0)

        

        axa6.errorbar(ind, W_sst_pv_means_afterreward[:]/nS, yerr=W_sst_pv_std_afterreward[:]/nS, color='.75', ls = ' ', marker='o', capsize=5, capthick=1, ecolor='black')


        axa6.set_ylabel('SST-to-PV weights [nS]')
        axa6.set_xlabel('SST population')
        axa6.set_xticks(ind)
        axa6.set_xticklabels(('|','/','--','\\'))
        axa6.set_ylim(-.02,1.02)
        axa6.set_xlim(0,N_pop)

        plt.tight_layout()

        plt.savefig('%s/%s/%s/2ndAfterreward.eps'%(savepath,dataname,run_no), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

        
       
            
        """Final """
        fig, ((axa1,axa3),(axa2,axa4),(ax6, ax5)) = plt.subplots(3,2,figsize=(6.3,8), gridspec_kw = {'width_ratios':[1.2, 1]})
        axa1.axvspan((nonplasticwarmup+plasticwarmup)*10,(nonplasticwarmup+plasticwarmup+rewardsimtime)*10, facecolor='.8',lw=0.0, alpha=0.5)
        
        tsplot(axa1,data=PYR1and2toothers[:,:]/nS,color = cmaps.viridis(1.0))#,label='others to others')
        tsplot(axa1,data=PYR0toothers[:,:]/nS, color = cmaps.viridis(0.0))#,label ='1 to others')
        tsplot(axa1,data=otherstoPYR0[:,:]/nS, color = cmaps.viridis(0.5))#,label='others to 1')

        axa1.set_xlabel('time [seconds]')
        axa1.set_ylabel('excitatory weights [nS]')
        axa1.set_xlim((nonplasticwarmup+plasticwarmup+rewardsimtime)*10,total*10)
        axa1.set_xticks(np.arange((nonplasticwarmup+plasticwarmup+rewardsimtime+2.1)*10,total*10,100))
        axa1.set_xticklabels(np.arange(70,int(total),10))

        
        tsplot(axa2,data=SOMPV_w[:3600,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
        tsplot(axa2,data=SOMPV_w[3600:7200,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
        tsplot(axa2,data=SOMPV_w[7200:10800,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
        tsplot(axa2,data=SOMPV_w[10800:14400,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')

        axa2.set_xlabel('time [seconds]')
        axa2.set_ylabel('SST-to-PV weights [nS]')
        axa2.set_ylim(-.02,1.02)
        axa2.set_xlim((nonplasticwarmup+plasticwarmup+rewardsimtime),total)
        axa2.set_xticks(np.arange((nonplasticwarmup+plasticwarmup+rewardsimtime+2.1),total,10))
        axa2.set_xticklabels(np.arange(70,int(total),10))


        plot_synapses(W_pyr_i, W_pyr_j, W_rec/gmax, var_name='normalised exciatory weights',
                           plot_type='scatter', cmap='viridis', rasterized=True, axes =  axa3)
        add_background_pattern(axa3)
        axa3.set_xlabel('presynaptic')
        axa3.set_xticks(np.arange(0,N_pyr+10,100))

        axa3.set_ylabel('postsynaptic')
        axa3.set_yticks(np.arange(0,N_pyr+10,100))

        
        
        ind = np.arange(N_pop)+.5 # the x locations for the groups
        width = 0.35       # the width of the bars
        groupsize = (N_sst*N_pv)/N_pop
        a = cmaps.viridis(0.0)
        

        axa4.errorbar(ind, W_sst_pv_means[i,:]/nS, yerr=W_sst_pv_std[i,:]/nS, color='.75', ls = ' ', marker='o', capsize=5, capthick=1, ecolor='black')


        x0,x1 = axa3.get_xlim()
        y0,y1 = axa3.get_ylim()
        axa4.set_aspect( 4.0/1.0 )

        axa4.set_ylabel('final SST-to-PV weights [nS]')
        axa4.set_xlabel('SST population')
        axa4.set_xticks(ind)
        axa4.set_xticklabels(('|','/','--','\\'))
        axa4.set_ylim(-.02,1.02)
        axa4.set_xlim(0,N_pop)

        if checker == True:
            image = plot_synapses(W_pyr_i, W_pyr_j, W_rec2/gmax, var_name='normalised excitatory weights',
                                  plot_type='scatter', cmap='magma', rasterized=True, axes=ax5)
        else: 
            image = plot_synapses(W_pyr_i, W_pyr_j, W_rec/gmax, var_name='normalised excitatory weights',
                                  plot_type='scatter', cmap='magma', rasterized=True, axes=ax5)


        add_background_pattern(ax5)

        ax5.set_xlabel('presynaptic')
        ax5.set_xticks(np.arange(0,N_pyr+10,100))

        ax5.set_ylabel('postsynaptic')
        ax5.set_yticks(np.arange(0,N_pyr+10,100))
            
        ax6.axvspan((nonplasticwarmup+plasticwarmup)*10,(nonplasticwarmup+plasticwarmup+rewardsimtime)*10, facecolor='.8',lw=0.0, alpha=0.5)
        if checker == True:
            PYR1and2toothers_2 = np.concatenate((PYR1toothers_2,PYR2toothers_2))

            tsplot(ax6,data=PYR1and2toothers_2[:,:]/nS,color = cmaps.viridis(1.0))#,label='others to others')
            tsplot(ax6,data=PYR0toothers_2[:,:]/nS, color = cmaps.viridis(0.0))#,label ='1 to others')
            tsplot(ax6,data=otherstoPYR0_2[:,:]/nS, color = cmaps.viridis(0.5))#,label='others to 1')
        else: 
            tsplot(ax6,data=PYR1and2toothers[:,:]/nS,color = cmaps.viridis(1.0))#,label='others to others')
            tsplot(ax6,data=PYR0toothers[:,:]/nS, color = cmaps.viridis(0.0))#,label ='1 to others')
            tsplot(ax6,data=otherstoPYR0[:,:]/nS, color = cmaps.viridis(0.5))#,label='others to 1')

        ax6.set_xlabel('time [seconds]')
        ax6.set_ylabel('excitatory weight [nS]')
        ax6.set_xticks(np.arange(0,total*10,200))
        ax6.set_xticklabels(np.arange(0,int(total),20))
        lgd = plt.legend(['others to others','1 to others','others to 1'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
        
        plt.tight_layout()

        
        plt.savefig('%s/%s/%s/2ndFinal.eps'%(savepath,dataname,run_no), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

        first_response = tunings_before
        last_response = tunings_after


        

        tuning_before, tuning_after = get_pop_tuning(tunings_before[:,:4],tunings_after[:,:4])
        tuning_initial[i,:,:], tuning_final[i,:,:] = get_pop_tuning(tunings_initial[:,:4],tunings_final[:,:4])
        SSTtuning_initial[i,:,:], SSTtuning_final[i,:,:] = get_pop_tuning(SOMtunings_initial[:,:4],SOMtunings_final[:,:4])
        PVtuning_initial[i,:,:], PVtuning_final[i,:,:] = get_pop_tuning(PVtunings_initial[:,:4],PVtunings_final[:,:4])
        #VIPtuning_initial[i,:,:], VIPtuning_final[i,:,:] = get_pop_tuning(VIPtunings_initial[:,:4],VIPtunings_final[:,:4])

       
    
        tuning_after_norm = np.zeros((no_stimuli,no_stimuli))


        plot_tuningcurves(tunings_initial[:,:4],None,None,tunings_final[:,:4],N_pyr,N_pop,save='%s/%s/%s'%(savepath, dataname, run_no),name = '')
        
        plot_tuningcurves(SOMtunings_initial,None,None,SOMtunings_final, N_sst,N_pop,save='%s/%s/%s'%(savepath, dataname, run_no),name = 'SST')
        plot_tuningcurves(PVtunings_initial,PVtuning_rewardend,PVtuning_after_rewardend,PVtunings_final, N_pv,1,save='%s/%s/%s'%(savepath, dataname, run_no),name = 'PVall')
        plot_tuningcurves(PVtunings_initial,None,None,PVtunings_final, N_pv,1,save='%s/%s/%s'%(savepath, dataname, run_no),name = 'PV')
        plot_tuningcurves(PVtunings_initial,None,PVtuning_after_rewardend,None, N_pv,1,save='%s/%s/%s'%(savepath, dataname, run_no),name = 'PVrewardend')
        
        
        plot_tuningcurves(tunings_initial,tunings_afterwarmup,tuning_after_rewardend,tunings_final,N_pyr,N_pop,save='%s/%s/%s'%(savepath, dataname, run_no),name = 'warmup')
        if SOMswitch == True:
            plot_tuningcurves(tunings_final[:,:4],None,None,tuning_finalSOMswitch[:,:4],N_pyr,N_pop,save='%s/%s/%s'%(savepath, dataname, run_no),name = 'finalSOMswitch')
            plot_SOMswitch(tunings_final[:,:4],tuning_finalSOMswitch[:,:4],N_pyr,N_pop,save='%s/%s/%s'%(savepath, dataname, run_no),name = 'final')
                
        plot_alltuningcurves(tunings_initial[:,:4],None,None,tunings_final[:,:4],N_pyr,N_pop,save='%s/%s/%s'%(savepath, dataname, run_no),name = '')
        
        plot_alltuningcurves(SOMtunings_initial,None,None,SOMtunings_final, N_sst,N_pop,save='%s/%s/%s'%(savepath, dataname, run_no),name = 'SST')
       
        plot_alltuningcurves(PVtunings_initial,None,None,PVtunings_final, N_pv,1,save='%s/%s/%s'%(savepath, dataname, run_no),name = 'PV')
        plot_alltuningcurves(PVtunings_initial,None,PVtuning_after_rewardend,None, N_pv,1,save='%s/%s/%s'%(savepath, dataname, run_no),name = 'PV')
        



        #Big figure
        fig, ((axa1,axa3),(axa2,axa4)) = plt.subplots(2,2)#,figsize=(15,15))
        axa1.axvspan((nonplasticwarmup+plasticwarmup)*10,(nonplasticwarmup+plasticwarmup+rewardsimtime)*10, facecolor='.8',lw=0.0, alpha=0.5)
        tsplot(axa1,data=PYR1and2toothers[:,:]/nS,color = cmaps.viridis(1.0))#,label='others to others')
        tsplot(axa1,data=PYR0toothers[:,:]/nS, color = cmaps.viridis(0.0))#,label ='1 to others')
        tsplot(axa1,data=otherstoPYR0[:,:]/nS, color = cmaps.viridis(0.5))#,label='others to 1')
        axa1.set_xlabel('time [seconds]')
        axa1.set_ylabel('excitatory weights [nS]')
        axa1.set_xticks(np.arange(0,total*10,100))
        axa1.set_xticklabels(np.arange(0,int(total),10))

        
        axa2.axvspan((nonplasticwarmup+plasticwarmup),(nonplasticwarmup+plasticwarmup+rewardsimtime), facecolor='.8',lw=0.0, alpha=0.5)
        tsplot(axa2,data=SOMPV_w[:3600,:]/nS,color = cmaps.viridis(0.0))#,label='others to others')
        tsplot(axa2,data=SOMPV_w[3600:7200,:]/nS,color = cmaps.viridis(.33))#,label ='1 to others')
        tsplot(axa2,data=SOMPV_w[7200:10800,:]/nS,color = cmaps.viridis(.66))#,label ='1 to others')
        tsplot(axa2,data=SOMPV_w[10800:14400,:]/nS, color = cmaps.viridis(1.0))#,label ='1 to others')

        axa2.set_xlabel('time [seconds]')
        axa2.set_ylabel('SST-to-PV weights [nS]')
        axa2.set_ylim(-.02,1.02)
        axa2.set_xticks(np.arange(0,total,20))
        axa2.set_xticklabels(np.arange(0,int(total),20))
       

        plot_synapses(W_pyr_i, W_pyr_j, W_rec/gmax, var_name='normalised exciatory weights',
                           plot_type='scatter', cmap='magma', rasterized=True, axes =  axa3)
        add_background_pattern(axa3)
        axa3.set_xlabel('presynaptic')
        axa3.set_xticks(np.arange(0,N_pyr+10,100))

        axa3.set_ylabel('postsynaptic')
        axa3.set_yticks(np.arange(0,N_pyr+10,100))

        
        ind = np.arange(N_pop)+.5 # the x locations for the groups
        width = 0.35       # the width of the bars
        groupsize = (N_sst*N_pv)/N_pop
        a = cmaps.viridis(0.0)
        
        plot_responsechangehistogram(tunings_initial,tunings_final, N_pyr,N_pop,axa4,save=None,name = '')

        plt.tight_layout()
        plt.savefig('%s/%s/%s/bigfigure.eps'%(savepath,dataname,run_no), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

        
        plot_tuningcurves(tunings_initial,tuning_rewardend,tuning_after_rewardend,tunings_final,N_pyr,N_pop,save='%s/%s/%s'%(savepath, dataname, run_no),name = '')
        plot_tuningcurves(SOMtunings_initial,SOMtuning_rewardend,SOMtuning_after_rewardend,SOMtunings_final, N_sst,N_pop,save='%s/%s/%s'%(savepath, dataname, run_no),name = 'SST')
        plot_tuningcurves(PVtunings_initial,PVtuning_rewardend,PVtuning_after_rewardend,PVtunings_final, N_pv,1,save='%s/%s/%s'%(savepath, dataname, run_no),name = 'PV')


             

        if xcorr == True:
            binsize = 5
    
            # during reward
            PYRData0_reward = results['PYRData0_reward']
            PYRData1_reward = results['PYRData1_reward']
            SSTData_reward = results['SSTData_reward']
            PVData_reward = results['PVData_reward']
            SSTData1_reward = results['SSTData1_reward']
            PVData1_reward = results['PVData1_reward']
    
            ## during stimulus 0
            
            PYR1train = np.hstack(PYRData0_reward['10'])*1000
            PYR2train = np.hstack(PYRData0_reward['11'])*1000
            PYR3train = np.hstack(PYRData0_reward['111'])*1000

    
            PVtrain = np.hstack(PVData_reward['10'])*1000
            SSTtrain = np.hstack(SSTData_reward['10'])*1000
            rewardstart = int(1000*(nonplasticwarmup+plasticwarmup))

    
            rewardend = int(1000*(nonplasticwarmup+plasticwarmup+rewardsimtime))
            end = int(1000*(nonplasticwarmup+plasticwarmup+rewardsimtime+norewardsimtime))
            sst_reward, binssst, patches = plt.hist(SSTtrain, bins=range(rewardstart,rewardend,binsize))
            pv_reward, binspv, patches = plt.hist(PVtrain, bins=range(rewardstart,rewardend,binsize))
            pyr1_reward, binspyr1, patches = plt.hist(PYR1train, bins=range(rewardstart,rewardend,binsize))
            pyr2_reward, binspyr2, patches = plt.hist(PYR2train, bins=range(rewardstart,rewardend,binsize))
            pyr3_reward, binspyr3, patches = plt.hist(PYR3train, bins=range(rewardstart,rewardend,binsize))
    
            ## during stimulus 1
            PYR1train1 = np.hstack(PYRData1_reward['10'])*1000
            PYR2train1 = np.hstack(PYRData1_reward['11'])*1000
            PYR3train1 = np.hstack(PYRData1_reward['111'])*1000
            PVtrain1 = np.hstack(PVData1_reward['10'])*1000
            SSTtrain1 = np.hstack(SSTData1_reward['10'])*1000
            pyr1_reward1, binspyr2, patches = plt.hist(PYR1train1, bins=range(rewardstart,rewardend,binsize))
            pyr2_reward1, binspyr2, patches = plt.hist(PYR2train1, bins=range(rewardstart,rewardend,binsize))
            pyr3_reward1, binspyr3, patches = plt.hist(PYR3train1, bins=range(rewardstart,rewardend,binsize))
            sst_reward1, binssst, patches = plt.hist(SSTtrain1, bins=range(rewardstart,rewardend,binsize))
            pv_reward1, binspv, patches = plt.hist(PVtrain1, bins=range(rewardstart,rewardend,binsize))
            
            # after reward
            PYRData0 = results['PYRData0']
            PYRData1 = results['PYRData1']
            SSTData = results['SSTData']
            SSTData1 = results['SSTData1']
            PVData = results['PVData']
            PVData1 = results['PVData1']
    
            ## during stimulus 0
            PYR1train1 = np.hstack(PYRData0['10'])*1000
            PYR2train1 = np.hstack(PYRData0['11'])*1000
            PYR3train1 = np.hstack(PYRData0['111'])*1000
            PVtrain1 = np.hstack(PVData['10'])*1000
            SSTtrain1 = np.hstack(SSTData['10'])*1000
            sst_afterreward, binssst, patches = plt.hist(SSTtrain1, bins=range(rewardend,end,binsize))
            pv_afterreward, binspv, patches = plt.hist(PVtrain1, bins=range(rewardend,end,binsize))
            pyr1_afterreward, binspyr1, patches = plt.hist(PYR1train1, bins=range(rewardend,end,binsize))
            pyr2_afterreward, binspyr2, patches = plt.hist(PYR2train1, bins=range(rewardend,end,binsize))
            pyr3_afterreward, binspyr3, patches = plt.hist(PYR3train1, bins=range(rewardend,end,binsize))
    
            ## during stimulus 1
            PYR1train1 = np.hstack(PYRData1['10'])*1000
            PYR2train1 = np.hstack(PYRData1['11'])*1000
            PYR3train1 = np.hstack(PYRData1['111'])*1000
            PVtrain1 = np.hstack(PVData1['10'])*1000
            SSTtrain1 = np.hstack(SSTData1['10'])*1000
            sst_afterreward1, binssst, patches = plt.hist(SSTtrain1, bins=range(rewardend,end,binsize))
            pv_afterreward1, binspv, patches = plt.hist(PVtrain1, bins=range(rewardend,end,binsize))
            pyr1_afterreward1, binspyr1, patches = plt.hist(PYR1train1, bins=range(rewardend,end,binsize))
            pyr2_afterreward1, binspyr2, patches = plt.hist(PYR2train1, bins=range(rewardend,end,binsize))
            pyr3_afterreward1, binspyr3, patches = plt.hist(PYR3train1, bins=range(rewardend,end,binsize))
    
            maxlags = 10
        

            plot_xcorr(pyr1_reward,pyr3_reward, sst_reward, pv_reward, 'reward')
            plot_xcorr2(pyr1_reward,pyr3_reward, sst_reward, pv_reward, 'reward2')
            plot_xcorr(pyr1_afterreward,pyr3_afterreward, sst_afterreward, pv_afterreward, 'afterreward')
            plot_xcorr2(pyr1_afterreward,pyr3_afterreward, sst_afterreward, pv_afterreward, 'afterreward2')



    # plot excitatory structure index as a function of varied_param
    plt.figure(figsize=(3.5,3))
    for pa in np.unique(dep_param2):
        plt.plot(dep_param[dep_param2==pa]/nS, impact_afterreward[dep_param2==pa], '.',markersize = 10, label=str(pa))#, c=performance_binary)
    plt.ylabel('Excitatory structure index')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
    plt.xlabel(varied_param)    

    plt.savefig('%s/%s/%s/impact_afterreward_colored.eps'%(savepath,dataname,run_no), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

    plt.figure(figsize=(3.5,3))
    for pa in np.unique(dep_param2):
        plt.plot(dep_param[dep_param2==pa]/nS, impact[dep_param2==pa], '.',markersize = 10, label=str(pa))#, c=performance_binary)
    plt.ylabel('Excitatory structure index')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)

    plt.xlabel(varied_param)    

    plt.savefig('%s/%s/%s/impact_colored.eps'%(savepath,dataname,run_no), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

    plt.figure(figsize=(3.5,3))
    for pa in np.unique(dep_param):
        plt.plot(dep_param2[dep_param==pa]/nS, impact_afterreward[dep_param==pa], '.',markersize = 10, label=str(pa))#, c=performance_binary)
    plt.ylabel('Excitatory structure index')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
    plt.xlabel(varied_param2)    

    plt.savefig('%s/%s/%s/impact_afterrewardp2_colored.eps'%(savepath,dataname,run_no), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

    plt.figure(figsize=(3.5,3))
    for pa in np.unique(dep_param):
        plt.plot(dep_param2[dep_param==pa]/nS, impact[dep_param==pa], '.',markersize = 10, label=str(pa))#, c=performance_binary)
    plt.ylabel('Excitatory structure index')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False, framealpha=1)
    plt.xlabel(varied_param2)    

    plt.savefig('%s/%s/%s/impactp2_colored.eps'%(savepath,dataname,run_no), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 


  

    W_sst0_pv = W_sst_pv_means[0]
    W_sstother_pv = np.mean(W_sst_pv[:,30:],1)
    plt.figure(figsize=(3.5,3))
    plt.plot(dep_param/nS, impact, '.',color='k', markersize = 5)#, c=performance_binary)
    plt.ylabel('Excitatory structure index')
    plt.xlabel('SST-to-PYR synaptic strength [nS]')

    plt.savefig('%s/%s/%s/impact.eps'%(savepath,dataname,run_no), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

    plt.figure(figsize=(3.5,3))
    plt.plot(dep_param[(dep_param/10**(-9))>0.3]/nS, impact_afterreward[(dep_param/10**(-9))>0.3], '.',color='k', markersize = 10)
    plt.ylabel('Excitatory structure index')
    plt.xlabel('SST-to-PC synaptic strength [nS]')    
    plt.savefig('%s/%s/%s/impact_afterreward.eps'%(savepath,dataname,run_no), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 


    plt.figure(figsize=(3.5,3))
    plt.plot(dep_param/nS, impactmax, '.',color='k', markersize = 5)#, c=performance_binary)
    plt.ylabel('Excitatory structure index')
    plt.xlabel('SST-to-PYR synaptic strength [nS]')

    plt.savefig('%s/%s/%s/impactmax.eps'%(savepath,dataname,run_no), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 

    plt.figure(figsize=(3.5,3))
    plt.plot(dep_param/nS, impactmax_afterreward, '.',color='k', markersize = 5)#, c=performance_binary)

    plt.ylabel('Excitatory structure index')
    plt.xlabel('SST-to-PYR synaptic strength [nS]')    
    plt.savefig('%s/%s/%s/impactmax_afterreward.eps'%(savepath,dataname,run_no), bbox_extra_artists=(lgd,), bbox_inches='tight',format='pdf', transparent=True) 


    counter = 0
    param_range = np.unique(dep_param)
    param_range2 = np.unique(dep_param2)
    impact_m = np.zeros((len(param_range),len(param_range2)))
    sst_pv_m = np.zeros((len(param_range),len(param_range2)))
    tuning_m = np.zeros((len(param_range),len(param_range2)))

    for c1, par1 in enumerate(param_range):
        for c2, par2 in enumerate(param_range2):
            try:
                impact_m[c1,c2] = impact[((dep_param == par1) & (dep_param2 == par2))]
                sst_pv_m[c1,c2] = sst_pv[((dep_param == par1) & (dep_param2 == par2))]
                tuning_m[c1,c2] = tuning[((dep_param == par1) & (dep_param2 == par2))]

            except ValueError:
                print('excepted ValueError')
          


    measures = {'impact':impact_m, 
                'sst_pv':sst_pv_m,
                'tuning increase':tuning_m,

                }

    for key in measures:          
        plt.figure()
        ax = subplot(111)
        plt.imshow(measures[key], interpolation='nearest', origin='lower')#, c=performance_binary)
        plt.xlabel(varied_param2)
        plt.ylabel(varied_param)
        plt.xticks(np.arange(len(param_range2)),param_range2)
        plt.yticks(np.arange(len(param_range)),param_range)
        cb = plt.colorbar(orientation='horizontal')
        cb.set_label(key)    

        plt.savefig('%s/%s/%s/%s3D_%s.eps'%(savepath,dataname,run_no,key,varied_param)) 

    fig, ax = plt.subplots(figsize=(5,4))
    ax.errorbar(ind, np.mean(W_sst_pv_means,0)/nS, yerr=np.mean(W_sst_pv_std,0)/nS, color='0.75', ls = ' ', marker='o', capsize=5, capthick=1, ecolor='black')


    ax.set_ylabel('final connection strength [nS]')
    ax.set_xlabel('SST population')

    ax.set_xticklabels((' ','|','/','--','\\'))
    ax.set_ylim(bottom=0)
    ax.set_xlim(-1,N_pop)

    plt.savefig('%s/%s/%s/SOMPV_dist_grandavg.eps'%(savepath, dataname, run_no)) 

    
    plot_grandavgtuningcurves(tuning_initial,tuning_final, N_pop, save='%s/%s/%s'%(savepath, dataname, run_no),name = '')
    plot_grandavgtuningcurves(SSTtuning_initial,SSTtuning_final, N_pop, save='%s/%s/%s'%(savepath, dataname, run_no),name = 'SST')
    plot_grandavgtuningcurves(PVtuning_initial,PVtuning_final, 1, save='%s/%s/%s'%(savepath, dataname, run_no),name = 'PV')

    

