#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
L2/3: PCs, PVs, SOMs and VIPs receive L4 bottom-up and top-down input

Created on Mon Mar  6 14:12:15 2017

@author: kwilmes
"""
import os
import shutil
from tempfile import mkdtemp
import numpy as np
import pickle


from brian2 import *
from brian2tools import *
from sacred import Experiment
ex = Experiment("L23_network") 

from analyse_experiment import *
from plot_Spikingmodel import *


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
class TmpExpDir(object):
        """A context manager that creates and deletes temporary directories.
        """

        def __init__(self, base_dir="./"):
                self._base_dir = base_dir
                self._exp_dir = None
        
        def __enter__(self):
                # create a temporary directory into which we will store all our files
                # it will be placed into the current directory but you could change that
                self._exp_dir = mkdtemp(dir=self._base_dir)
                return self._exp_dir

        def __exit__(self, *args):
                # at the very end of the run delete the temporary directory
                # sacred will have taken care of copying all the results files over
                # to the run directoy
                if self._exp_dir is not None:
                        shutil.rmtree(self._exp_dir)


def calc_impact(con_REC):
    pop1impact = np.mean(con_REC['i<100 and j>100'])
    otherimpact = (np.mean(con_REC['i>100 and i<300 and j>300'])+
                  np.mean(con_REC['i>200 and j>100 and j<200'])+
                  np.mean(con_REC['i>100 and i<200 and j>200 and j<300'])+
                  np.mean(con_REC['i>300 and j>200 and j<300']))/4
    self_impact = (np.mean(con_REC['i<100 and j<100'])+
                   np.mean(con_REC['i>100 and i<200 and j>100 and j<200'])+
                   np.mean(con_REC['i>200 and i<300 and j>200 and j<300'])+
                   np.mean(con_REC['i>300 and i<400 and j>300 and j<400']))/4
    impact_normtoself = (pop1impact - otherimpact)/self_impact
    impact_normtomax = (pop1impact - otherimpact)/np.max(con_REC)
    print("impact")

    return impact_normtoself, impact_normtomax 


# function that defines parameters of the model:
@ex.config
def config():

    
    
    params = {
        # simulation parameters
        'plot': False, 				# enables plotting during the run
        'seed' : 7472, 				# random seed       
	'nonplasticwarmup_simtime' : 1.4*second,# no plasticity, to measure tuning
        'warmup_simtime' : 1.4*second,# 42*second, 		# plasticity, no reward
        'reward_simtime' : 1.4*second,# 24.5*second, 	# plasticity, with reward
        'noreward_simtime': 1.4*second,# 45*second, 		# plasticity, without reward
        'noSSTPV_simtime': 1.4*second,# 21*second, 		# plasticity, without reward 
	# for Suppl. Figure, we killed SSTPV structure after 45s, therefore the no reward simtime is split up
        'after_simtime' : 1.4*second, 		# no plasticity, to measure tuning
        'timestep' : 0.1*ms,
    
        # number of neurons
        'NPYR' : 400,          	# Number of excitatory L23 PYR cells
        'NSOM' : 30*4,         	# Number of inhibitory SOM cells
        'NVIP' : 50,          	# Number of inhibitory VIP cells
        'NPV' : 120,		# Number of inhibitory PV cells
        'NTD' : 100,		# Number of top-down units
        
        
        # time constants of synaptic kernels
        'tau_ampa' : 5.0*ms,   	# Excitatory synaptic time constant
        'tau_gaba' : 10.0*ms,  	# Inhibitory synaptic time constant
        
        # L4 input
        'N4' : 4,  		# Number of L4 units
        'L4_rate' : 4/(1*ms), 	# Firing rate of L4 units
        'orientations' : np.array([0.785398163397, 1.57079632679, 2.35619449019, 0.0]), # Four orientations
        'input_time' : 70*ms, 	# stim_time + gap_time, i.e. time between starts of two subsequent stimuli
        'stim_time' : 50*ms, 	# duration of stimulus
        
        # L23 neuron parameters
        'gl' : 10.0*nsiemens,   # Leak conductance
        'el' : -60*mV,          # Resting potential
        'er' : -80*mV,          # Inhibitory reversal potential
        'vt' : -50.*mV,         # Spiking threshold
        'memc' : 200.0*pfarad,  # Membrane capacitance 
        'sigma' : 2.0*mV, 	# sigma of Ornstein-Uhlenbeck noise
        'tau_noise' : 5.0*ms, 	# tau of Ornstein-Uhlenbeck noise
        
	# Connectivity

	#p_pre_post is the probability of connection from a neuron in the presynaptic population to a neuron in the postsynaptic population
	#w_pre_post is the synaptic strength of the connection from a neuron in the presynaptic population to a neuron in the postsynaptic population

        'p_PYR_PYR' : 1.0, 
        'recurrent_weights' : 'clip(.01 * randn() + .01, 0, .15)*nS', 	# initial weights between PCs
        
        'p_SOM_PV' : .857,
        'SOM2PV_weights' : 'clip(.1 * randn() + .2, 0, 1.0)*nS', 	# initial weights between SST and PV

        'p_L4_TD' : 1.0,
        'p_TD_VIP' : 1.0,


        'p_PYR_SOM' : 1.0,
        'p_PYR_VIP' : 1.0,
        'p_PYR_PV' : .88,
        'p_SOM_PYR' : 1.0,
        'p_SOM_VIP' : 1.0,
        'p_PV_PV' : 1.0,
        'p_VIP_SOM' : 1.0,
        'p_VIP_PYR' : .125,
        'p_VIP_PV' : .125,
        'p_PV_SOM' : .125,
        'p_PV_PYR' : 1.0,
        'p_PV_VIP' : 1.0,
        'p_VIP_VIP' : .125,
        'p_SOM_SOM' : .125,

        'w_PYR_SOM' : 0.07*nS,
        'w_PYR_VIP' : 0.07*nS,
        'w_PYR_PV' : .12*nS,
        'w_SOM_PYR' : 0.3*nS,
        'w_SOM_VIP' : 0.42*nS,
        'w_PV_PV' : 0.55*nS,
        'w_VIP_SOM' : 0.195*nS,
        'w_PV_SOM' : 0.08*nS,
        'w_PV_PYR' : 0.55*nS,
        'w_PV_VIP' : 0.12*nS,
        'w_VIP_PYR' : .0675*nS,
        'w_VIP_PV' : .0675*nS,
        'w_VIP_VIP' : .0*nS,
        'w_SOM_SOM' : .0675*nS,
        'w_L4PYR' : .28*nS,
        'w_FFPYR' : .13*nS,
        'w_FFPV' : .01*nS,
        'w_FFSOM' : .15*nS,
        'w_TDVIP' : .2*nS,

	# gap junction parameters
        'w_gap' : 0*nS, 	# sub-threshold coupling
        'c_gap' : 13*pA, 	# spikelet current
        'tau_spikelet' : 9.0*ms,# spikelet time constant
        
        
        # Plasticity parameters
        'tau_stdp' : 20*ms, 	# STDP time constant at excitatory synapses
        'tau_istdp' : 20*ms, 	# STDP time constant at inhibitory synapses
        'dApre' : .005, 	# STDP amplitude
        'dApre_i' : 0.015, 	# Inhibitory STDP amplitude
        'gmax' : .25*nS, 	# maximum synaptic weight for excitatory synapses       
        'gmax_SSTPV': 1.0*nS, 	# maximum synaptic weight for SST-to-PV synapses
        'relbound' : .1, 	# maximum synaptic weight bound relative to initial weight

        'restplastic' : False, 	# if True all connections are plastic


    }



@ex.command
def run_network(params,_run):

    # get parameters
    p = Struct(**params)

    # simulation
    total_simtime = p.nonplasticwarmup_simtime + p.warmup_simtime + p.reward_simtime + p.noreward_simtime + p.noSSTPV_simtime + p.after_simtime
    total_warmup_simtime = p.nonplasticwarmup_simtime + p.warmup_simtime
    stim_time = p.stim_time    
    input_time = p.input_time
    seed(p.seed)
    
    # neurons
    N4 = p.N4
    L4_rate = p.L4_rate
    gl = p.gl
    el = p.el
    er = p.er
    vt = p.vt
    memc = p.memc
    tau_gaba = p.tau_gaba    
    tau_ampa = p.tau_ampa
    tau = p.tau_noise
    sigma = p.sigma

    # connections
    w_PYR_PV = p.w_PYR_PV
    w_PYR_VIP = p.w_PYR_VIP
    w_PYR_SOM = p.w_PYR_SOM
    w_FFPYR = p.w_FFPYR
    w_FFPV = p.w_FFPV
    w_FFSOM = p.w_FFSOM
    w_TDVIP = p.w_TDVIP
    w_L4PYR = p.w_L4PYR

    c_gap = p.c_gap
    tau_spikelet = p.tau_spikelet    
    
    # plasticity
    tau_stdp = p.tau_stdp 
    tau_istdp = p.tau_istdp 
    
    relbound = p.relbound
    gmax_SSTPV = p.gmax_SSTPV
    
    dApre = p.dApre*nS
    dApost = -dApre * tau_stdp / tau_stdp * 1.05
    
    dApre_i = p.dApre_i*nS    
    dApost_i = -dApre_i * tau_istdp / tau_istdp * 1.05
        

    # untuned Layer 4 neurons:
    eqs_FF = '''
    rate = L4_rate: Hz
    '''
    FF = NeuronGroup(1, eqs_FF, threshold='rand() < rate*dt',
                         method='euler', name='FF')
    
    # tuned Layer 4 neurons:*((t<(stim_end_time+10*ms)))
    eqs_layer4 = '''
    rate = clip(cos(orientation*2 - selectivity*2), 0, inf)*L4_rate : Hz
    stim_rate = rate*(t<stim_end_time): Hz
    gap_rate = (L4_rate*2/5)*(t>=stim_end_time) : Hz
    selectivity : 1 			# preferred orientation
    orientation : 1 (shared) 		# orientation of the current stimulus
    stim_start_time : second (shared) 	# start time of the current stimulus
    stim_end_time : second (shared) 	# end time of the current stimulus
    '''
    layer4 = NeuronGroup(N4, eqs_layer4, threshold='rand() < stim_rate *dt',
                         method='euler', name='layer4')
    gapfiller = NeuronGroup(N4, '''gap_rate : Hz (linked)''', threshold='rand() < gap_rate *dt',
                         method='euler', name='gapfiller')
    gapfiller.gap_rate = linked_var(layer4, 'gap_rate')
    
    # selectivities for N4 = 4 neurons: 180, 45, 90, and 135 degrees in radians
    layer4.selectivity = '(i%N4)/(1.0*N4)*pi'  # for each L4 neuron, selectivity between 0 and pi

    # Choose one of the four preferred oriented bars every 70ms (discrete stimulus)
    # idx = int(floor(rand()*N4)) for N4=4 samples uniformly from [0,1,2,3]
    # orientation = (idx%4)/(1.0*4)*pi 
    runner_code = '''
    orientation = ((int(floor(rand()*N4)))%4)/(1.0*4)*pi 
    stim_start_time = t
    stim_end_time = t + stim_time
    '''
    layer4.run_regularly(runner_code, dt=p.input_time, when='start')

    Stimmonitor = SpikeMonitor(layer4, variables=['orientation'])

    # L23 neurons        
    
    eqs_neurons='''
    dv/dt=(-gl*(v-el)+Isyn+Igap+Ispikelet)/memc + sigma * (2 / tau)**.5 *xi: volt (unless refractory)
    Isyn = IsynE + IsynI : amp
    IsynE = -g_ampa*v : amp
    IsynI = -g_gaba*(v-er) : amp
    Igap: amp
    dIspikelet/dt = -Ispikelet/tau_spikelet : amp
    dg_ampa/dt = -g_ampa/tau_ampa : siemens
    dg_gaba/dt = -g_gaba/tau_gaba : siemens
    '''
    
    # Excitatory synapses
    STDP_E = '''w : siemens
        	gmax : siemens
                dApre/dt = -Apre / tau_stdp : siemens (event-driven)
                dApost/dt = -Apost / tau_stdp : siemens (event-driven)
                plastic : boolean (shared)
                '''
    # STDP at excitatory synapses
    on_pre_STDP_E = '''g_ampa += w
    			Apre += dApre
    			w = clip(w + plastic*Apost, 0, gmax)'''

    on_post_STDP_E = '''Apost += dApost
    			w = clip(w + plastic*Apre, 0, gmax)'''

    # anti-Hebbian STDP at excitatory synapses
    on_pre_antiHebb_IE = '''g_ampa += w
                       	Apre += dApre
                       	w = clip(w - plastic*Apost, 0, gmax)'''
    
    on_post_antiHebb_IE = '''Apost += dApost
                       	w = clip(w - plastic*Apre, 0, gmax)'''

 

    # define neurons
    exc_neurons = NeuronGroup(p.NPYR, model=eqs_neurons, threshold='v > vt',
                          reset='v=el', refractory=2*ms, method='euler')

    inh_neurons = NeuronGroup(p.NSOM+p.NVIP+p.NPV, model=eqs_neurons, threshold='v > vt',
                          reset='v=el', refractory=2*ms, method='euler')    
    
    PYR = exc_neurons[:p.NPYR]
    SOM = inh_neurons[:p.NSOM]
    VIP = inh_neurons[p.NSOM:int(p.NSOM+p.NVIP)]
    PV = inh_neurons[int(p.NSOM+p.NVIP):]

    neuron1 = StateMonitor(PYR, ('v', 'Isyn', 'IsynE', 'IsynI'), record=PYR[0:1])

    neuron2 = StateMonitor(PYR, ('v','Isyn','IsynE','IsynI'), record=PYR[100:101])
    neuron3 = StateMonitor(PYR, ('v','IsynE', 'IsynI'), record=PYR[200:201])
    neuron4 = StateMonitor(PYR, ('v','IsynE','IsynI'), record=PYR[300:301])

    currents = StateMonitor(PYR, ('IsynE', 'IsynI'), record=[0,1,2,3,4,100,101,102,103,104,200,201,202,203,204,300,301,302,303,304])

    SOMneuron1 = StateMonitor(SOM, ('v','Isyn','IsynE','IsynI'), record=SOM[0:1])
    SOMneuron2 = StateMonitor(SOM, ('v','Isyn','IsynE','IsynI'), record=SOM[30:31])

    VIPneuron1 = StateMonitor(VIP, ('v','Isyn','IsynE','IsynI'), record=VIP[0:1])
    PVneuron1 = StateMonitor(PV, ('v','Isyn','IsynE','IsynI'), record=PV[0:1])
    
    # Feedforward synaptic connections from L4 to L23
    
    feedforward1 = Synapses(layer4[0:1], PYR[0:100], 
                            '''w = w_L4PYR: siemens''',
                            on_pre='g_ampa += w', name='feedforward1')
    feedforward1.connect(p=1)
    feedforward2 = Synapses(layer4[1:2], PYR[100:200], on_pre='g_ampa += w_L4PYR', name='feedforward2')
    feedforward2.connect(p=1)
    feedforward3 = Synapses(layer4[2:3], PYR[200:300], on_pre='g_ampa += w_L4PYR', name='feedforward3')
    feedforward3.connect(p=1)
    feedforward4 = Synapses(layer4[3:4], PYR[300:400], on_pre='g_ampa += w_L4PYR', name='feedforward4')
    feedforward4.connect(p=1)

    feedforwardgap1 = Synapses(gapfiller[0:1], PYR[0:100], on_pre='g_ampa += w_L4PYR', name='feedforwardgap1')
    feedforwardgap1.connect(p=1)
    feedforwardgap2 = Synapses(gapfiller[1:2], PYR[100:200], on_pre='g_ampa += w_L4PYR', name='feedforwardgap2')
    feedforwardgap2.connect(p=1)
    feedforwardgap3 = Synapses(gapfiller[2:3], PYR[200:300], on_pre='g_ampa += w_L4PYR', name='feedforwardgap3')
    feedforwardgap3.connect(p=1)
    feedforwardgap4 = Synapses(gapfiller[3:4], PYR[300:400], on_pre='g_ampa += w_L4PYR', name='feedforwardgap4')
    feedforwardgap4.connect(p=1)

    feedforward_unspec = Synapses(FF, PYR, on_pre='g_ampa += w_FFPYR', name='feedforward_unspec')
    feedforward_unspec.connect(p=1)

    feedforward_PV = Synapses(FF, PV, on_pre='g_ampa += w_FFPV', name='feedforward_PV')
    feedforward_PV.connect(p=1)


    feedforward_i1 = Synapses(layer4[0:1], SOM[0:30], on_pre='g_ampa += w_FFSOM', name='feedforward_i1')
    feedforward_i1.connect(p=1)
    feedforward_i2 = Synapses(layer4[1:2], SOM[30:60], on_pre='g_ampa += w_FFSOM', name='feedforward_i2')
    feedforward_i2.connect(p=1)
    feedforward_i3 = Synapses(layer4[2:3], SOM[60:90], on_pre='g_ampa += w_FFSOM', name='feedforward_i3')
    feedforward_i3.connect(p=1)
    feedforward_i4 = Synapses(layer4[3:4], SOM[90:120], on_pre='g_ampa += w_FFSOM', name='feedforward_i4')
    feedforward_i4.connect(p=1)
    feedforward_gap1 = Synapses(gapfiller[0:1], SOM[0:30], on_pre='g_ampa += w_FFSOM*1.1', name='feedforward_gapi1')
    feedforward_gap1.connect(p=1)
    feedforward_gap2 = Synapses(gapfiller[1:2], SOM[30:60], on_pre='g_ampa += w_FFSOM*1.1', name='feedforward_gapi2')
    feedforward_gap2.connect(p=1)
    feedforward_gap3 = Synapses(gapfiller[2:3], SOM[60:90], on_pre='g_ampa += w_FFSOM*1.1', name='feedforward_gapi3')
    feedforward_gap3.connect(p=1)
    feedforward_gap4 = Synapses(gapfiller[3:4], SOM[90:120], on_pre='g_ampa += w_FFSOM*1.1', name='feedforward_gapi4')
    feedforward_gap4.connect(p=1)
        
    # Synaptic connections within L23

    # Connections from PCs to SSTs:
    on_pre_PCSOM = on_pre_antiHebb_IE
    on_post_PCSOM = on_post_antiHebb_IE
        
    PYR_SOM1 = Synapses(PYR[0:100], SOM[0:30], STDP_E, on_pre= on_pre_PCSOM,on_post = on_post_PCSOM,name='PYR_SOM1')
    PYR_SOM1.connect(p=p.p_PYR_SOM)
    PYR_SOM1.w = w_PYR_SOM
    PYR_SOM1.gmax = w_PYR_SOM+relbound*nS

    PYR_SOM2 = Synapses(PYR[100:200], SOM[30:60], STDP_E, on_pre= on_pre_PCSOM,on_post = on_post_PCSOM, name='PYR_SOM2')
    PYR_SOM2.connect(p=p.p_PYR_SOM)
    PYR_SOM2.w = w_PYR_SOM
    PYR_SOM2.gmax = w_PYR_SOM+relbound*nS

    PYR_SOM3 = Synapses(PYR[200:300], SOM[60:90], STDP_E, on_pre= on_pre_PCSOM,on_post = on_post_PCSOM, name='PYR_SOM3')
    PYR_SOM3.connect(p=p.p_PYR_SOM)
    PYR_SOM3.w = w_PYR_SOM
    PYR_SOM3.gmax = w_PYR_SOM+relbound*nS

    PYR_SOM4 = Synapses(PYR[300:400], SOM[90:120], STDP_E, on_pre= on_pre_PCSOM,on_post = on_post_PCSOM, name='PYR_SOM4')
    PYR_SOM4.connect(p=p.p_PYR_SOM)        
    PYR_SOM4.w = w_PYR_SOM
    PYR_SOM4.gmax = w_PYR_SOM+relbound*nS

    # Inhibitory synapses
    Synaptic_model_I = '''w : siemens
            gmax_i : siemens
            dApre_i/dt = -Apre_i / tau_istdp : siemens (event-driven)
            dApost_i/dt = -Apost_i / tau_istdp : siemens (event-driven)
            plastic : boolean (shared)'''
    
    # STDP at inhibitory synapses
    on_pre_STDP_I = '''g_gaba += w
                       Apre_i += dApre_i
                       w = clip(w + plastic*Apost_i, 0, gmax_i)'''
    
    on_post_STDP_I = '''Apost_i += dApost_i
                       w = clip(w + plastic*Apre_i, 0, gmax_i)'''


    # anti-Hebbian STDP at inhibitory synapses
    on_pre_antiHebb_I = '''g_gaba += w
                       Apre_i += dApre_i
                       w = clip(w - plastic*Apost_i, 0, gmax_i)'''
    
    on_post_antiHebb_I = '''Apost_i += dApost_i
                       w = clip(w - plastic*Apre_i, 0, gmax_i)'''

    
    
    """excitatory synapses"""
    # plastic recurrent synapses
    con_REC = Synapses(PYR, PYR,
                       STDP_E,
                       on_pre=on_pre_STDP_E,
                       on_post=on_post_STDP_E,
                       name='recurrent')
    con_REC.connect(p=p.p_PYR_PYR)
    con_REC.gmax = p.gmax

    con_REC.w = p.recurrent_weights
    
    # SST to PV
    con_SOM_PV = Synapses(SOM, PV,
                       Synaptic_model_I,
                       on_pre=on_pre_STDP_I,
                       on_post=on_post_STDP_I,
                       name='som2pv')

    con_SOM_PV.connect(p=p.p_SOM_PV)
    con_SOM_PV.w = p.SOM2PV_weights
    con_SOM_PV.gmax_i = p.gmax_SSTPV
    

    # PYR to PV 
    con_PYR_PV = Synapses(PYR, PV, STDP_E,
                          on_pre= on_pre_STDP_E, 
                          on_post = on_post_STDP_E,
                          name='PYR0_PV0')
    con_PYR_PV.connect(p=p.p_PYR_PV)
    con_PYR_PV.w = w_PYR_PV
    con_PYR_PV.gmax = p.w_PYR_PV+relbound*nS

    # PC to VIP
    con_PYR_VIP = Synapses(PYR, VIP, STDP_E,
                          on_pre= on_pre_STDP_E, 
                          on_post = on_post_STDP_E,
                          name='PYR0_VIP0')
    con_PYR_VIP.connect(p=p.p_PYR_VIP)
    con_PYR_VIP.w = w_PYR_VIP
    con_PYR_VIP.gmax = p.w_PYR_VIP+relbound*nS


    """inhibitory synapses"""       
    # SST to PC
    con_SOM_PYR = Synapses(SOM, PYR, Synaptic_model_I, 
                          on_pre=on_pre_STDP_I,
                          on_post=on_post_STDP_I,
                          name='SOMPYR')
    con_SOM_PYR.connect(p=p.p_SOM_PYR)
    con_SOM_PYR.w = p.w_SOM_PYR
    con_SOM_PYR.gmax_i = p.w_SOM_PYR+relbound*nS
    
    # SST to VIP
    con_SOM_VIP = Synapses(SOM, VIP, Synaptic_model_I, 
                          on_pre='''g_gaba += w
                       		Apre_i += dApre_i
                       		w = clip(w + plastic*.1*Apost_i, 0, gmax_i)''',
                          on_post='''Apost_i += dApost_i
                       		w = clip(w + plastic*.1*Apre_i, 0, gmax_i)''',
                          name='SOMVIP')
    con_SOM_VIP.connect(p=p.p_SOM_VIP)
    con_SOM_VIP.w = p.w_SOM_VIP
    con_SOM_VIP.gmax_i = p.w_SOM_VIP+relbound*nS

    #SST to SST
    con_SOM_SOM = Synapses(SOM, SOM, Synaptic_model_I, 
                          on_pre=on_pre_STDP_I,
                          on_post=on_post_STDP_I,
                          name='SOMSOM')
    con_SOM_SOM.connect(p=p.p_SOM_SOM)
    con_SOM_SOM.w = p.w_SOM_SOM
    con_SOM_SOM.gmax_i = p.w_SOM_SOM+relbound*nS


    # PV to PC
    con_PV_PYR = Synapses(PV, PYR,Synaptic_model_I, 
                          on_pre=on_pre_antiHebb_I,
                          on_post=on_post_antiHebb_I,
                          name='PVPYR')
    con_PV_PYR.connect(p=p.p_PV_PYR)
    con_PV_PYR.w = p.w_PV_PYR
    con_PV_PYR.gmax_i = p.w_PV_PYR+relbound*nS

    #PV to SST
    con_PV_SOM = Synapses(PV, SOM,Synaptic_model_I, 
                          on_pre=on_pre_STDP_I,
                          on_post=on_post_STDP_I,
                          name='PVSOM')
    con_PV_SOM.connect(p=p.p_PV_SOM)
    con_PV_SOM.w = p.w_PV_SOM
    con_PV_SOM.gmax_i = p.w_PV_SOM+relbound*nS

    #PV to VIP
    con_PV_VIP = Synapses(PV, VIP,Synaptic_model_I, 
                          on_pre=on_pre_STDP_I,
                          on_post=on_post_STDP_I,
                          name='PVVIP')
    con_PV_VIP.connect(p=p.p_PV_VIP)
    con_PV_VIP.w = p.w_PV_VIP
    con_PV_VIP.gmax_i = p.w_PV_VIP+relbound*nS

    #PV to PV
    con_PV_PV = Synapses(PV, PV,Synaptic_model_I, 
                          on_pre=on_pre_STDP_I,
                          on_post=on_post_STDP_I,
                          name='PVPV')
    con_PV_PV.connect(p=p.p_PV_PV)
    con_PV_PV.w = p.w_PV_PV
    con_PV_PV.gmax_i = p.w_PV_PV+relbound*nS
    
    
    # VIP to SST

    on_pre_VIPSOM = on_pre_antiHebb_I
    on_post_VIPSOM = on_post_antiHebb_I    
    
    con_VIP_SOM = Synapses(VIP, SOM, Synaptic_model_I, 
                           on_pre=on_pre_VIPSOM,
                           on_post=on_post_VIPSOM,
                           name='VIPSOM')
    con_VIP_SOM.connect(p=p.p_VIP_SOM)
    con_VIP_SOM.w = p.w_VIP_SOM
    con_VIP_SOM.gmax_i = p.w_VIP_SOM+relbound*nS

    # VIP to PC
    con_VIP_PYR = Synapses(VIP, PYR,Synaptic_model_I, 
                          on_pre=on_pre_STDP_I,
                          on_post=on_post_STDP_I,
                          name='VIPPYR')
    con_VIP_PYR.connect(p=p.p_VIP_PYR)
    con_VIP_PYR.w = p.w_VIP_PYR
    con_VIP_PYR.gmax_i = p.w_VIP_PYR+relbound*nS

    # VIP to PV
    con_VIP_PV = Synapses(VIP, PV,Synaptic_model_I, 
                          on_pre=on_pre_STDP_I,
                          on_post=on_post_STDP_I,
                          name='VIPPV')
    con_VIP_PV.connect(p=p.p_VIP_PV)
    con_VIP_PV.w = p.w_VIP_PV
    con_VIP_PV.gmax_i = p.w_VIP_PV+relbound*nS

    # VIP to VIP
    con_VIP_VIP = Synapses(VIP, VIP,Synaptic_model_I, 
                          on_pre=on_pre_STDP_I,
                          on_post=on_post_STDP_I,
                          name='VIPVIP')
    con_VIP_VIP.connect(p=p.p_VIP_VIP)
    con_VIP_VIP.w = p.w_VIP_VIP
    con_VIP_VIP.gmax_i = p.w_VIP_VIP+relbound*nS
       
    # gap junctions between PVs
    PVPV_gap = Synapses(PV, PV, '''w : siemens
                        Igap_post = w * (v_pre - v_post) : amp (summed)
                        ''',
                        on_pre='Ispikelet+=c_gap',
                        )
    PVPV_gap.connect()
    PVPV_gap.w = p.w_gap
    
    # monitor synaptic weights  
    monPYRPV = StateMonitor(con_PYR_PV, 'w', record=True, dt = 1000*ms)
    monVIPSOM = StateMonitor(con_VIP_SOM, 'w', record=True, dt = 1000*ms)
    monVIPPV = StateMonitor(con_VIP_PV, 'w', record=True, dt = 1000*ms)
    monVIPPYR = StateMonitor(con_VIP_PYR, 'w', record=True, dt = 1000*ms)
    monPVPYR = StateMonitor(con_PV_PYR, 'w', record=True, dt = 1000*ms)
    monPVSOM = StateMonitor(con_PV_SOM, 'w', record=True, dt = 1000*ms)
    monPVPV = StateMonitor(con_PV_PV, 'w', record=True, dt = 1000*ms)
    monPVVIP = StateMonitor(con_PV_VIP, 'w', record=True, dt = 1000*ms)
    monSOMVIP = StateMonitor(con_SOM_VIP, 'w', record=True, dt = 1000*ms)
    monSOMPYR = StateMonitor(con_SOM_PYR, 'w', record=True, dt = 1000*ms)
    monSOMSOM = StateMonitor(con_SOM_SOM, 'w', record=True, dt = 1000*ms)
    monPYRSOM1 = StateMonitor(PYR_SOM1, 'w', record=True, dt = 1000*ms)
    monPYRSOM2 = StateMonitor(PYR_SOM2, 'w', record=True, dt = 1000*ms)
    monPYRSOM3 = StateMonitor(PYR_SOM3, 'w', record=True, dt = 1000*ms)
    monPYRSOM4 = StateMonitor(PYR_SOM4, 'w', record=True, dt = 1000*ms)
    monPYRVIP = StateMonitor(con_PYR_VIP, 'w', record=True, dt = 1000*ms)
    monVIPVIP = StateMonitor(con_VIP_VIP, 'w', record=True, dt = 1000*ms)

    # monitor excitatory connections
    mona = StateMonitor(con_REC, 'w', record=con_REC[0:100,100:400], dt = 100*ms) # pyr 1 to others
    monb = StateMonitor(con_REC, 'w', record=con_REC[100:400,0:100], dt = 100*ms) # other to pyr 1
    monc = StateMonitor(con_REC, 'w', record=con_REC[100:200,200:400], dt = 100*ms) # pyr2 to others
    mond = StateMonitor(con_REC, 'w', record=con_REC[300:400,100:300], dt = 100*ms) # pyr 4 to others
    mone = StateMonitor(con_REC, 'w', record=con_REC[0:100,0:100], dt = 100*ms) # pyr 1 to pyr1
    monf = StateMonitor(con_REC, 'w', record=con_REC[100:200,100:200], dt = 100*ms) # pyr 1 to pyr1
    

    # monitor population rates
    PYR1 = PopulationRateMonitor(PYR[0:100])
    PYR2 = PopulationRateMonitor(PYR[100:200])
    PYR3 = PopulationRateMonitor(PYR[200:300])
    PYR4 = PopulationRateMonitor(PYR[300:400])
    SOM1 = PopulationRateMonitor(SOM[0:30])
    SOM2 = PopulationRateMonitor(SOM[30:60])
    SOM3 = PopulationRateMonitor(SOM[60:90])
    SOM4 = PopulationRateMonitor(SOM[90:120])
    PVmon = PopulationRateMonitor(PV)
    VIPmon = PopulationRateMonitor(VIP)

    # monitor SST to PV connections
    monSOMPV = StateMonitor(con_SOM_PV, 'w', record=True, dt = 1000*ms)
    SOM0PV = StateMonitor(con_SOM_PV, 'w', record=con_SOM_PV[:30:10,::40])
    SOMotherPV = StateMonitor(con_SOM_PV, 'w', record=con_SOM_PV[30::10,1::40])


    # Top down input: reward for stimulus 0 (horizontal, 180 degrees)
    TD = NeuronGroup(p.NTD, model=eqs_neurons, threshold='v > vt',
                          reset='v=el', refractory=2*ms, method='euler')
    
    con_ff_td = Synapses(layer4[0:1], TD, on_pre='g_ampa += 0.3*nS')
    con_ff_td.connect(p=p.p_L4_TD)
    
    # top down input goes onto VIP
    con_topdown = Synapses(TD, VIP, on_pre='g_ampa += w_TDVIP')
    con_topdown.connect(p=p.p_TD_VIP)

    # monitor spikes
    sm_PYR = SpikeMonitor(PYR)
    sm_VIP = SpikeMonitor(VIP)
    sm_SOM = SpikeMonitor(SOM)
    sm_PV = SpikeMonitor(PV)
    sm_TD = SpikeMonitor(TD)    
    sm_layer4 = SpikeMonitor(layer4)
    sm_FF = SpikeMonitor(FF)
    sm_gap = SpikeMonitor(gapfiller)
    
    

    # run without plasticity
    defaultclock.dt = p.timestep

    con_ff_td.active = False
    TD.active = False  
    con_REC.plastic = False
    con_SOM_PV.plastic = False
    con_PYR_PV.plastic = False
    PYR_SOM1.plastic = False
    PYR_SOM2.plastic = False
    PYR_SOM3.plastic = False
    PYR_SOM4.plastic = False
    con_PYR_VIP.plastic = False
    con_VIP_SOM.plastic = False
    con_VIP_PV.plastic = False
    con_VIP_VIP.plastic = False
    con_VIP_PYR.plastic = False
    con_SOM_PYR.plastic = False
    con_SOM_VIP.plastic = False
    con_SOM_SOM.plastic = False
    con_PV_SOM.plastic = False
    con_PV_PYR.plastic = False
    con_PV_VIP.plastic = False
    con_PV_PV.plastic = False

    conREC_start = np.copy(con_REC.w[:])
    run(p.nonplasticwarmup_simtime, report = 'text')
    store('nonplasticwarmup')
    print('non-plastic warmup done')


    # plastic warmup
    restore('nonplasticwarmup')
    con_ff_td.active = False
    TD.active = False  
    con_REC.plastic = True
    con_SOM_PV.plastic = True
    
    if p.restplastic == True:
        con_VIP_SOM.plastic = True
        con_PYR_PV.plastic = True
        con_PV_PYR.plastic = True
        con_PYR_VIP.plastic = True
        PYR_SOM1.plastic = True
        PYR_SOM2.plastic = True
        PYR_SOM3.plastic = True
        PYR_SOM4.plastic = True
        con_VIP_PV.plastic = True
        con_VIP_VIP.plastic = True
        con_VIP_PYR.plastic = True
        con_SOM_PYR.plastic = True
        con_SOM_VIP.plastic = True
        con_SOM_SOM.plastic = True
        con_PV_SOM.plastic = True
        con_PV_VIP.plastic = True
        con_PV_PV.plastic = True
    else:
        con_PYR_PV.plastic = False
        PYR_SOM1.plastic = False
        PYR_SOM2.plastic = False
        PYR_SOM3.plastic = False
        PYR_SOM4.plastic = False
        con_PYR_VIP.plastic = False
        con_VIP_SOM.plastic = False
        con_VIP_PV.plastic = False
        con_VIP_VIP.plastic = False
        con_VIP_PYR.plastic = False
        con_SOM_PYR.plastic = False
        con_SOM_VIP.plastic = False
        con_SOM_SOM.plastic = False
        con_PV_SOM.plastic = False
        con_PV_PYR.plastic = False
        con_PV_VIP.plastic = False
        con_PV_PV.plastic = False

    print('starting warmup')
    run(p.warmup_simtime, report = 'text')
    conREC_afterwarmup = np.copy(con_REC.w[:])
    sstpv_w_afterwarmup = np.copy(con_SOM_PV.w[:])
    store('afterwarmup')
    print('warmup done')


    # rewarded phase        
    restore('afterwarmup')    
    con_ff_td.active = True
    TD.active = True  
    con_REC.plastic = True
    con_SOM_PV.plastic = True
    print('starting reward period')
    run(p.reward_simtime, report = 'text')
    impact_afterreward, impactmax_afterreward = calc_impact(con_REC.w)
    print('calculated impacts')
    conREC_afterreward = np.copy(con_REC.w[:])
    print('copied con_Rec')
    sstpv_w_afterreward = np.copy(con_SOM_PV.w[:])
    print('copied sstpv')
    store('afterreward')
    print('rewarded phase done')

    # refinement phase
    restore('afterreward')    
    con_SOM_PV.plastic = True
    con_ff_td.active = False
    con_topdown.active = False
    TD.active = False    

    print('starting refinement phase')
    run(p.noreward_simtime, report = 'text')
    store('afternoreward')
    print('45s of refinement phase done')
    
    

    # refinement phase, option to kill SST-PV structure
    restore('afternoreward')    
    # For Suppl. Fig. kill inhibitory weight structure:
    # con_SOM_PV.w_i = p.SOM2PV_weights
    con_ff_td.active = False
    TD.active = False    
    con_REC.plastic = True
    con_SOM_PV.plastic = True
    run(p.noSSTPV_simtime, report = 'text')
    store('afternoSSTPV')
    print('refinement phase done')
    
    # final non-plastic phase to measure tuning
    restore('afternoSSTPV')    
    con_ff_td.active = False
    TD.active = False    
    con_REC.plastic = False
    con_SOM_PV.plastic = False
    con_PYR_PV.plastic = False
    PYR_SOM1.plastic = False
    PYR_SOM2.plastic = False
    PYR_SOM3.plastic = False
    PYR_SOM4.plastic = False    
    con_PYR_VIP.plastic = False
    con_VIP_SOM.plastic = False
    con_VIP_PV.plastic = False
    con_VIP_VIP.plastic = False
    con_VIP_PYR.plastic = False
    con_SOM_PYR.plastic = False
    con_SOM_VIP.plastic = False
    con_SOM_SOM.plastic = False
    con_PV_SOM.plastic = False
    con_PV_PYR.plastic = False
    con_PV_VIP.plastic = False
    con_PV_PV.plastic = False
    run(p.after_simtime)    

    # get spiking information
    PYR_spiketrains = sm_PYR.spike_trains()
    SOM_spiketrains = sm_SOM.spike_trains()
    VIP_spiketrains = sm_VIP.spike_trains()
    PV_spiketrains = sm_PV.spike_trains()
    stimuli_t = Stimmonitor.t
    
    PYRi, PYRt = sm_PYR.it
    SSTi, SSTt = sm_SOM.it
    PVi, PVt = sm_PV.it
    VIPi, VIPt = sm_VIP.it
    gapi, gapt = sm_gap.it
    
    results = {
        'SOM0PV' : SOM0PV.w,
        'SOMotherPV' : SOMotherPV.w,
        'weights_rec' : con_REC.w[:],
        'weights_rec_afterwarmup' : conREC_afterwarmup,
        'weights_rec_afterreward' : conREC_afterreward,
        'weights_rec_start' : conREC_start,
        'weights_rec_i' : con_REC.i[:],
        'weights_rec_j' : con_REC.j[:],
        'weights_sst_pv': con_SOM_PV.w[:],
        'weights_sst_pv_afterreward' : sstpv_w_afterreward, 
        'weights_sst_pv_afterwarmup' : sstpv_w_afterwarmup, 
        't' : PYR1.t[:],
        'SOMPV_t' : monSOMPV.t[:],
        'SOMPV_w' : monSOMPV.w[:], 
        'SOMPV_t' : monSOMPV.t[:],
        'SOMPV_w' : monSOMPV.w[:], 
        'PYRPV_w' : monPYRPV.w[:], 
        'PYRVIP_w' : monPYRVIP.w[:], 
        'PVPYR_w' : monPVPYR.w[:], 
        'PVPV_w' : monPVPV.w[:], 
        'PVSOM_w' : monPVSOM.w[:], 
        'PVVIP_w' : monPVVIP.w[:], 
        'VIPSOM_w' : monVIPSOM.w[:], 
        'VIPPYR_w' : monVIPPYR.w[:], 
        'VIPPV_w' : monVIPPV.w[:], 
        'VIPVIP_w' : monVIPVIP.w[:], 
        'SOMVIP_w' : monSOMVIP.w[:], 
        'SOMPYR_w' : monSOMPYR.w[:], 
        'SOMSOM_w' : monSOMSOM.w[:], 
        'PYRSOM1_w' : monPYRSOM1.w[:], 
        'PYRSOM2_w' : monPYRSOM2.w[:], 
        'PYRSOM3_w' : monPYRSOM3.w[:], 
        'PYRSOM4_w' : monPYRSOM4.w[:],           
        'PYR0toothers': mona.w,
        'otherstoPYR0': monb.w,
        'PYR1toothers': monc.w,
        'PYR2toothers': mond.w,        
        'PYRi' : PYRi[:],
        'PYRt' : PYRt[:],
        'SSTi' : SSTi[:],
        'SSTt' : SSTt[:],
        'PVi' : PVi[:],
        'PVt' : PVt[:],
        'VIPi' : VIPi[:],
        'VIPt' : VIPt[:],
        'Pyr1rate' : PYR1.smooth_rate(window='flat', width=0.5*ms),
        'Pyr2rate' : PYR2.smooth_rate(window='flat', width=0.5*ms),
        'Pyr3rate' : PYR3.smooth_rate(window='flat', width=0.5*ms),
        'Pyr4rate' : PYR4.smooth_rate(window='flat', width=0.5*ms),
        'SOM1rate' : SOM1.smooth_rate(window='flat', width=0.5*ms),
        'SOM2rate' : SOM2.smooth_rate(window='flat', width=0.5*ms),
        'SOM3rate' : SOM3.smooth_rate(window='flat', width=0.5*ms),
        'SOM4rate' : SOM4.smooth_rate(window='flat', width=0.5*ms),
        'PVrate' : PVmon.smooth_rate(window='flat', width=0.5*ms),
    

    }
    
    # create a temporary directory into which we will store all files
    # it will be placed into the current directory but this can be changed
    # this temporary directory will automatically be deleted as soon as the with statement ends
    with TmpExpDir(base_dir="./") as exp_dir:
            # lets create a filename for storing some data        
            results_file = os.path.join(exp_dir, "results.pkl")

            with open(results_file, 'wb') as f:
                    pickle.dump(results, f)

            # add the result as an artifact, note that the name here is important
            # as sacred otherwise will try to save to the oddly named tmp subdirectory we created
            ex.add_artifact(results_file, name=os.path.basename(results_file))
    


    _run.info["output"] = np.ones(3)
    
    # Data postprocessing

    # calculate impact of pyr0 onto others in weight matrix
    impact, impactmax = calc_impact(con_REC.w)

    
    PVrate_initial = get_firingrate(PV_spiketrains, 0*second, p.nonplasticwarmup_simtime)
    PVrate_TD = get_firingrate(PV_spiketrains, total_warmup_simtime, total_warmup_simtime + p.reward_simtime)
    
    no_stimuli = 4
    # get tuning for all populations to first and last presentation of each stimulus in entire simulation:
    tuning_before, tuning_after = get_tuning(PYR_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli)
    firstSOM, lastSOM = get_tuning(SOM_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli)
    firstVIP, lastVIP = get_tuning(VIP_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli)
    firstPV, lastPV = get_tuning(PV_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli)
    
    
    reward_endtime = total_warmup_simtime+p.reward_simtime#/p.timestep
    # get times of all stimuli during particular phases of the simulation:
    # in the very beginning (first), endofreward, startofnonreward, and at the very end (last)
    first, endofreward, startofnonreward, last = get_particular_stimulus_times(Stimmonitor.orientation, Stimmonitor.t, no_stimuli, reward_endtime, reward_endtime)

    tuning_rewardend = get_spike_response(PYR_spiketrains, no_stimuli, p.input_time, last=endofreward)
    tuning_after_rewardend = get_spike_response(PYR_spiketrains, no_stimuli, p.input_time, first=startofnonreward)
    
    # get tuning average over all stimulus presentations over a period of time
    tuning_initial = get_tuning_avgoverperiod(PYR_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=p.input_time, upto=p.nonplasticwarmup_simtime)
    tuning_afterwarmup = get_tuning_avgoverperiod(PYR_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime-p.nonplasticwarmup_simtime, upto=total_warmup_simtime)
    tuning_duringreward = get_tuning_avgoverperiod(PYR_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime+p.reward_simtime-p.nonplasticwarmup_simtime, upto=total_warmup_simtime+p.reward_simtime)
    tuning_afterreward = get_tuning_avgoverperiod(PYR_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime+p.reward_simtime, upto=total_warmup_simtime+p.reward_simtime+p.nonplasticwarmup_simtime)
    tuning_final = get_tuning_avgoverperiod(PYR_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_simtime - p.after_simtime, upto=total_simtime)
    
    stimtuning_initial = get_tuning_avgoverperiod(PYR_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, stim_time, startat=p.input_time, upto=p.nonplasticwarmup_simtime)
    stimtuning_final = get_tuning_avgoverperiod(PYR_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, stim_time, startat=total_simtime - p.after_simtime, upto=total_simtime)
    stimPVtuning_initial = get_tuning_avgoverperiod(PV_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, stim_time, startat=p.input_time, upto=p.nonplasticwarmup_simtime)
    stimPVtuning_final = get_tuning_avgoverperiod(PV_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, stim_time, startat=total_simtime - p.after_simtime, upto=total_simtime)
    
    
    PVtuning_initial = get_tuning_avgoverperiod(PV_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=p.input_time, upto=p.nonplasticwarmup_simtime)
    PVtuning_afterwarmup = get_tuning_avgoverperiod(PV_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime-p.nonplasticwarmup_simtime, upto=total_warmup_simtime)
    PVtuning_duringreward = get_tuning_avgoverperiod(PV_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime+p.reward_simtime-p.nonplasticwarmup_simtime, upto=total_warmup_simtime+p.reward_simtime)
    PVtuning_afterreward = get_tuning_avgoverperiod(PV_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime+p.reward_simtime, upto=total_warmup_simtime+p.reward_simtime+p.nonplasticwarmup_simtime)
    PVtuning_final = get_tuning_avgoverperiod(PV_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_simtime - p.after_simtime, upto=total_simtime)

    VIPtuning_initial = get_tuning_avgoverperiod(VIP_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=p.input_time, upto=p.nonplasticwarmup_simtime)
    VIPtuning_afterwarmup = get_tuning_avgoverperiod(VIP_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime-p.nonplasticwarmup_simtime, upto=total_warmup_simtime)
    VIPtuning_duringreward = get_tuning_avgoverperiod(VIP_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime+p.reward_simtime-p.nonplasticwarmup_simtime, upto=total_warmup_simtime+p.reward_simtime)
    VIPtuning_afterreward = get_tuning_avgoverperiod(VIP_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime+p.reward_simtime, upto=total_warmup_simtime+p.reward_simtime+p.nonplasticwarmup_simtime)
    VIPtuning_final = get_tuning_avgoverperiod(VIP_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_simtime - p.after_simtime, upto=total_simtime)


    SOMtuning_initial = get_tuning_avgoverperiod(SOM_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=p.input_time, upto=p.nonplasticwarmup_simtime)
    SOMtuning_afterwarmup = get_tuning_avgoverperiod(SOM_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime-p.nonplasticwarmup_simtime, upto=total_warmup_simtime)
    SOMtuning_duringreward = get_tuning_avgoverperiod(SOM_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime+p.reward_simtime-p.nonplasticwarmup_simtime, upto=total_warmup_simtime+p.reward_simtime)
    SOMtuning_afterreward = get_tuning_avgoverperiod(SOM_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime+p.reward_simtime, upto=total_warmup_simtime+p.reward_simtime+p.nonplasticwarmup_simtime)
    SOMtuning_final = get_tuning_avgoverperiod(SOM_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_simtime - p.after_simtime, upto=total_simtime)
  
    
    PYRData_reward = get_spiketrains_foreachstim(PYR_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime, upto=total_warmup_simtime+p.reward_simtime)
    SSTData_reward = get_spiketrains_foreachstim(SOM_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime, upto=total_warmup_simtime+p.reward_simtime)
    PVData_reward = get_spiketrains_foreachstim(PV_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime, upto=total_warmup_simtime+p.reward_simtime)
    PYRData_afterreward = get_spiketrains_foreachstim(PYR_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime+p.reward_simtime,upto=total_simtime - p.after_simtime)
    SSTData_afterreward = get_spiketrains_foreachstim(SOM_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime+p.reward_simtime,upto=total_simtime - p.after_simtime)
    PVData_afterreward = get_spiketrains_foreachstim(PV_spiketrains, Stimmonitor.orientation, Stimmonitor.t, no_stimuli, p.input_time, startat=total_warmup_simtime+p.reward_simtime,upto=total_simtime - p.after_simtime)

    try:
        currentratio_initial, currentratiomean_initial, ampE_initial, ampI_initial, amp2Ei, amp2Ii, amp3Ei, amp3Ii = get_currentratio_foreachstim(currents,Stimmonitor.orientation,Stimmonitor.t,no_stimuli, p.input_time,startat=total_warmup_simtime - p.nonplasticwarmup_simtime, upto=total_warmup_simtime)
        currentratio_final, currentratiomean_final, ampE_final, ampI_final, amp2Ef, amp2If, amp3Ef, amp3If = get_currentratio_foreachstim(currents,Stimmonitor.orientation,Stimmonitor.t,no_stimuli, p.input_time,startat=total_simtime - p.after_simtime, upto=total_simtime)
    except:
        currentratio_initial = []
        currentratiomean_initial = []
        ampE_initial = []
        ampI_initial = []
        amp2Ei = []
        amp2Ii = []
        amp3Ei = [] 
        amp3Ii = []
        currentratio_final = []
        currentratiomean_final = []
        ampE_final = []
        ampI_final = []
        amp2Ef = []
        amp2If = []
        amp3Ef = []
        amp3If = []

    
    results = {
        'impact': impact,
        'impact_aftereward': impactmax_afterreward,
        'impactmax': impact,
        'impactmax_aftereward': impactmax_afterreward,
        'SOM0PV' : SOM0PV.w,
        'SOMotherPV' : SOMotherPV.w,
        'weights_rec' : con_REC.w[:],
        'weights_rec_afterwarmup' : conREC_afterwarmup,
        'weights_rec_afterreward' : conREC_afterreward,
        'weights_rec_start' : conREC_start,
        'weights_rec_i' : con_REC.i[:],
        'weights_rec_j' : con_REC.j[:],
        'weights_sst_pv': con_SOM_PV.w[:],
        'weights_sst_pv_afterwarmup' : sstpv_w_afterwarmup, 
        'weights_sst_pv_afterreward' : sstpv_w_afterreward, 
        'stimtuning_initial': stimtuning_initial,
        'stimtuning_final': stimtuning_final,
        'stimPVtuning_initial': stimPVtuning_initial,
        'stimPVtuning_final': stimPVtuning_final,
        't' : PYR1.t[:],
        'tuning_initial': tuning_initial,
        'tuning_final': tuning_final,
        'tuning_afterwarmup': tuning_afterwarmup,
        'tuning_rewardend':tuning_duringreward,
        'tuning_after_rewardend':tuning_afterreward,
        'PVtuning_initial': PVtuning_initial,
        'PVtuning_final': PVtuning_final,
        'PVtuning_afterwarmup': PVtuning_afterwarmup,
        'PVtuning_rewardend':PVtuning_duringreward,
        'PVtuning_after_rewardend':PVtuning_afterreward,
        #'VIPtuning_initial': VIPtuning_initial,
        #'VIPtuning_final': VIPtuning_final,
        #'VIPtuning_afterwarmup': VIPtuning_afterwarmup,
        #'VIPtuning_rewardend':VIPtuning_duringreward,
        #'VIPtuning_after_rewardend':VIPtuning_afterreward,
        'SOMtuning_initial': SOMtuning_initial,
        'SOMtuning_final': SOMtuning_final,
        'SOMtuning_rewardend':SOMtuning_duringreward,
        'SOMtuning_after_rewardend':SOMtuning_afterreward,
        'SOMPV_t' : monSOMPV.t[:],
        'SOMPV_w' : monSOMPV.w[:], 
        'PYRPV_w' : monPYRPV.w[:], 
        #'PYRSOM_w' : monPYRSOM.w[:], 
        'PYRVIP_w' : monPYRVIP.w[:], 
        'PVPYR_w' : monPVPYR.w[:], 
        'PVPV_w' : monPVPV.w[:], 
        'PVSOM_w' : monPVSOM.w[:], 
        'PVVIP_w' : monPVVIP.w[:], 
        'VIPSOM_w' : monVIPSOM.w[:], 
        'VIPPYR_w' : monVIPPYR.w[:], 
        'VIPPV_w' : monVIPPV.w[:], 
        'VIPVIP_w' : monVIPVIP.w[:], 
        'SOMVIP_w' : monSOMVIP.w[:], 
        'SOMPYR_w' : monSOMPYR.w[:], 
        'SOMSOM_w' : monSOMSOM.w[:], 
        'PYRSOM1_w' : monPYRSOM1.w[:], 
        'PYRSOM2_w' : monPYRSOM2.w[:], 
        'PYRSOM3_w' : monPYRSOM3.w[:], 
        'PYRSOM4_w' : monPYRSOM4.w[:], 
        'currentratio_initial': currentratio_initial,
        'currentratio_final': currentratio_final,
        #'ampE_initial': ampE_initial,
        #'ampE_final': ampE_final,
        #'ampI_initial': ampI_initial,
        #'ampI_final': ampI_final,
        #'amp2E_initial': amp2Ei,
        #'amp2E_final': amp2Ef,
        #'amp2I_initial': amp2Ii,
        #'amp2I_final': amp2If,
        #'inh_currents1': neuron1.IsynI[0],
        #'exc_currents1': neuron1.IsynE[0],
        #'inh_currents2': neuron2.IsynI[0],
        #'exc_currents2': neuron2.IsynE[0],        
        #'inh_currents3': neuron3.IsynI[0],
        #'exc_currents3': neuron3.IsynE[0],          
        #'inh_currents4': neuron4.IsynI[0],
        #'exc_currents4': neuron4.IsynE[0],  
        #'inh_currentsPV': PVneuron1.IsynI[0],
        #'exc_currentsPV': PVneuron1.IsynE[0],  
        #'inh_currentsSOM1': SOMneuron1.IsynI[0],
        #'exc_currentsSOM1': SOMneuron1.IsynE[0],          
        #'inh_currentsSOM2': SOMneuron2.IsynI[0],
        #'exc_currentsSOM2': SOMneuron2.IsynE[0],          
        'PYR0toothers': mona.w,
        'otherstoPYR0': monb.w,
        'PYR1toothers': monc.w,
        'PYR2toothers': mond.w,        
        'PYR1toPYR1': mone.w,        
        'PYR2toPYR2': monf.w,
        'PYRi' : PYRi[:],
        'PYRt' : PYRt[:],
        'SSTi' : SSTi[:],
        'SSTt' : SSTt[:],
        'PVi' : PVi[:],
        'PVt' : PVt[:],
        'VIPi' : VIPi[:],
        'VIPt' : VIPt[:],
        'PYRData0_reward':PYRData_reward['0'], # during stimulus 0
        'PYRData1_reward':PYRData_reward['1'], # during stimulus 1
        'PVData_reward': PVData_reward['0'],
        'PVData1_reward': PVData_reward['1'],
        'SSTData_reward': SSTData_reward['0'],
        'SSTData1_reward': SSTData_reward['1'],
        'PYRData0':PYRData_afterreward['0'],
        'PYRData1':PYRData_afterreward['1'],
        'PVData': PVData_afterreward['0'],
        'PVData1': PVData_afterreward['1'], 
        'SSTData': SSTData_afterreward['0'],
        'SSTData1': SSTData_afterreward['1'],
        'Pyr1rate' : PYR1.smooth_rate(window='flat', width=0.5*ms),
        'Pyr2rate' : PYR2.smooth_rate(window='flat', width=0.5*ms),
        'Pyr3rate' : PYR3.smooth_rate(window='flat', width=0.5*ms),
        'Pyr4rate' : PYR4.smooth_rate(window='flat', width=0.5*ms),
        'SOM1rate' : SOM1.smooth_rate(window='flat', width=0.5*ms),
        'SOM2rate' : SOM2.smooth_rate(window='flat', width=0.5*ms),
        'SOM3rate' : SOM3.smooth_rate(window='flat', width=0.5*ms),
        'SOM4rate' : SOM4.smooth_rate(window='flat', width=0.5*ms),
        'PVrate' : PVmon.smooth_rate(window='flat', width=0.5*ms),
    

    }
    
    # create a temporary directory into which we will store all files
    # it will be placed into the current directory but this can be changed
    # this temporary directory will automatically be deleted as soon as the with statement ends
    with TmpExpDir(base_dir="./") as exp_dir:
            # lets create a filename for storing some data        
            results_file = os.path.join(exp_dir, "results.pkl")

            with open(results_file, 'wb') as f:
                    pickle.dump(results, f)

            # add the result as an artifact, note that the name here is important
            # as sacred otherwise will try to save to the oddly named tmp subdirectory we created
            ex.add_artifact(results_file, name=os.path.basename(results_file))
    
    
    
    
@ex.automain
def main(params):
    run_network()
    
    
    
