# some quick tools to help analyse your experiments

import glob
import os
import json
import pickle
from brian2 import *
from brian2tools import *
import matplotlib.pyplot as plt
import matplotlib.cm as cmaps
#import colormaps as cmaps

from sacred.serializer import restore

import numpy as np

class ExperimentReader(object):

    def __init__(self, run_dir):
        self._run_dir = run_dir

    @staticmethod
    def _load_json(f):
        pure = json.load(f)
        return pure#restore(pure) 

    @property
    def _files_in_rundir(self):
        return glob.glob('{}/*'.format(self._run_dir))
    
    def get_all_experiment_runs(self):
        # WARNING: this assumes that experiment do not start with _ 
        # (this should be the default sacred behavior though)
        results = {}
        for fname in self._files_in_rundir:
            # ignore directories starting with '_'
            if os.path.basename(fname)[0] != '_': 
                exp_id = os.path.basename(fname) 
                results[exp_id] = { 'run' : self.get_experiment_run(exp_id),
                                    'config' : self.get_experiment_config(exp_id)}#,
                                    #'info' : self.get_experiment_info(exp_id) }
        return results

    def get_experiment_config(self, exp_id):
        with open(os.path.join(self._run_dir, str(exp_id), 'config.json')) as f:
            return self._load_json(f)

    def get_experiment_run(self, exp_id):
        with open(os.path.join(self._run_dir, str(exp_id), 'run.json')) as f:
            return self._load_json(f)
            
    def get_experiment_info(self, exp_id):
        with open(os.path.join(self._run_dir, str(exp_id), 'info.json')) as f:
            return self._load_json(f)

    def get_experiment_stdout(self, exp_id):
        with open(os.path.join(self._run_dir, str(exp_id), 'cout.json')) as f:
            return f.readlines()

    def get_experiment_sterr(self, exp_id):
        with open(os.path.join(self._run_dir, str(exp_id), 'cerr.json')) as f:
            return f.readlines()

    def get_artifact_file_names(self, exp_id, run_dict=None):
        if run_dict is None:
            run_dict = self.get_experiment_run(exp_id)
        base_dir = os.path.join(self._run_dir, exp_id)
        return [os.path.join(base_dir, fname) for fname in run_dict['artifacts']]

    def try_loading_artifacts(self, exp_id, run_dict=None):
        if run_dict is None:
            run_dict = self.get_experiment_run(exp_id)
        base_dir = os.path.join(self._run_dir, exp_id)
        fnames = [os.path.join(base_dir, fname) for fname in run_dict['artifacts']]
        results = {}
        for fname in fnames:
            bname = os.path.basename(fname)
            extension = os.path.splitext(bname)[-1].lower()
            # try to guess how to read the artifact
            if extension in {'.pkl', '.pickle'}:
                with open(fname, 'rb') as f:
                    results[bname] = pickle.load(f)
            elif extension in {'.npy', '.np', '.npz'}:
                with open(fname) as f:
                    results[bname] = np.load(f)
            elif extension in {'.txt'}:
                results[bname] = np.loadtxt(fname)
            elif extension in {'.hdf5', 'h5'}:
                import h5py
                results[bname] = h5py.File(fname, 'r')
            elif extension in {'.json'}:
                with open(fname) as f:
                    results[bname] = self._load_json(f)
            else:
                raise ValueError("I have no clue how to load file {} with extension {}".format(fname, extension))
        return results

    def try_loading_artifacts_for_all_runs(self):
        runs = self.get_all_experiment_runs()
        res = {}
        for fname in self._files_in_rundir:
            if os.path.basename(fname)[0] != '_': 
                exp_id = os.path.basename(fname)
                res[exp_id] = self.try_loading_artifacts(exp_id)
        return res
                
    
def get_stimulus_times(stimuli_orientation, stimuli_t, no_stimuli):
    stimuli = np.zeros((np.shape(stimuli_orientation)[0]))
    orientations = np.unique(stimuli_orientation)
    # convert orientation to category 0,1,2,3
    for i in range(0,no_stimuli):
        stimuli[(stimuli_orientation==orientations[i])]=i

    # get time point of first and last presentation for each stimulus
    first = np.zeros(no_stimuli)
    last = np.zeros(no_stimuli)
    for i in range(no_stimuli):
        first[i] = stimuli_t[stimuli==i][1] # TEST
        last[i] = stimuli_t[stimuli==i][-1]

    return first, last    

def get_particular_stimulus_times(stimuli_orientation, stimuli_t, no_stimuli, upto = None, startat = None):
    stimuli = np.zeros((np.shape(stimuli_orientation)[0]))
    first = np.zeros((no_stimuli))
    endofsth = np.zeros((no_stimuli))
    startofsth = np.zeros((no_stimuli))
    last = np.zeros((no_stimuli)) 
    orientations = np.unique(stimuli_orientation)
    # convert orientation to category 0,1,2,3
    for i in range(0,no_stimuli):
        stimuli[(stimuli_orientation==orientations[i])]=i
    
    for i in range(no_stimuli):
        first[i] = stimuli_t[stimuli==i][0]
        if not upto == None:
            substimuli = stimuli[stimuli_t<upto]
            substimuli_t = stimuli_t[stimuli_t<upto]
            endofsth[i] = stimuli_t[substimuli==i][-1]
        if startat != None:
            if startat < stimuli_t[-1]:
                substimuli = stimuli[stimuli_t>startat]
                substimuli_t = stimuli_t[stimuli_t>startat]
                startofsth[i] = substimuli_t[substimuli==i][0]+startat
            else:
                raise ValueError('startat is not supposed to be at the end of the simulation')
        last[i] = stimuli_t[stimuli==i][-1]
    return first, endofsth, startofsth, last

def get_stimulus_sequence(stimuli_orientation, stimuli_t, simtime, input_time):
    stimulus_seq = []
    for j in range(0,int(simtime/ms),int(input_time/ms)):
        stimulus_seq.append(stimuli_orientation[(stimuli_t>j*ms)][0])
        
    return stimulus_seq

def get_spiketrains_foreachstim(spike_train, stimuli_orientation, stimuli_t, no_stimuli, input_time, stim_time=None, startat = None, upto = None):
    if stim_time == None:
        stim_time = input_time
    N_neurons = len(spike_train)
    orientations = np.unique(stimuli_orientation)
    if upto/second > np.max(stimuli_t):    
        upto = np.max(stimuli_t)*second
    # only take 
    startat_idx = np.nonzero(stimuli_t>startat)[0][0]
    response = np.zeros((N_neurons,no_stimuli))

    # get sequence of stimuli, which each lasts 50ms long
    stimulus_seq = get_stimulus_sequence(stimuli_orientation, stimuli_t, upto, input_time)
    # get index of stimulus in sequence that starts at time point startat:
    startat_idx = (startat/(input_time))
    
    Data = {}
    for i in range(0,no_stimuli):
        # get all indices of stimulus: 
        indices = np.nonzero(stimulus_seq == orientations[i])[0]
        # get only indices of stimuli that occur after time startat:
        indices = indices[indices>startat_idx]

        Neuron = {}
        for k in range(N_neurons):
            spike_trains = []            
            for idx in indices:
                # get spike counts of neuron k that happen during stimulus with index idx
                spike_trains.append((spike_train[k][(spike_train[k]>idx*input_time)&(spike_train[k]<idx*input_time+stim_time)]))
            Neuron[str(k)] = spike_trains
        Data[str(i)] = Neuron             
    return Data



def get_tuning_avgoverperiod(spike_train, stimuli_orientation, stimuli_t, no_stimuli, input_time, stim_time=None, startat = None, upto = None):
    if stim_time == None:
        stim_time = input_time
    N_neurons = len(spike_train)
    orientations = np.unique(stimuli_orientation)
    if upto/second > np.max(stimuli_t):    
        upto = np.max(stimuli_t)*second
    # only take 
    startat_idx = np.nonzero(stimuli_t>=startat)[0][0]
    print(startat_idx)
    response = np.zeros((N_neurons,no_stimuli))

    # get sequence of stimuli, which each lasts 50ms long
    stimulus_seq = get_stimulus_sequence(stimuli_orientation, stimuli_t, upto, input_time)
    # get index of stimulus in sequence that starts at time point startat:
    startat_idx = (startat/(input_time))

    for i in range(0,no_stimuli):
        # get all indices of stimulus: 
        indices = np.nonzero(stimulus_seq == orientations[i])[0]
        #print(indices)
        # get only indices of stimuli that occur after time startat:
        indices = indices[indices>=startat_idx]

        for k in range(N_neurons):
            spikes = []            
            for idx in indices:
                # get spike counts of neuron k that happen during stimulus with index idx
                spikes.append(np.sum(((spike_train[k]>=idx*input_time)&(spike_train[k]<(idx*input_time)+stim_time))))

            # mean number of spikes for all occurences of that stimulus
            response[k,i] = sum(spikes)/float(len(spikes)) 
    print(response)
    return response # multiply by correct number to get Hz (what is sampling frequency of Stimmonitor? should be 0.1ms)

def get_currentratio_foreachstim(currents, stimuli_orientation, stimuli_t, no_stimuli, input_time, stim_time=None, startat = None, upto = None):
    if stim_time == None:
        stim_time = input_time
    N_neurons = 20#len(currents)
    currentratio = np.zeros((N_neurons,no_stimuli))
    currentratiomean = np.zeros((N_neurons,no_stimuli))
    ampE = np.zeros((N_neurons,no_stimuli))
    ampI = np.zeros((N_neurons,no_stimuli))
    amp2E = np.zeros((N_neurons,no_stimuli))
    amp2I = np.zeros((N_neurons,no_stimuli))
    amp3E = np.zeros((N_neurons,no_stimuli))
    amp3I = np.zeros((N_neurons,no_stimuli))
    
    #print(len(currents))
    orientations = np.unique(stimuli_orientation)
    if upto/second > np.max(stimuli_t):    
        upto = np.max(stimuli_t)*second
    # only take 
    startat_idx = np.nonzero(stimuli_t>=startat)[0][0]

    #print('&&&&&&')
    #print(currents.t)
    # get sequence of stimuli, which each lasts 50ms long
    stimulus_seq = get_stimulus_sequence(stimuli_orientation, stimuli_t, upto, input_time)
    # get index of stimulus in sequence that starts at time point startat:
    #print(startat)
    startat_idx = (startat/(input_time))
    #print(startat_idx)
    #print(startat_idx*input_time+input_time)
    #print((currents.t>startat_idx*input_time)&(currents.t<startat_idx*input_time+input_time))
    #print(np.shape(currents.IsynI[0][:]))
    for i in range(0,no_stimuli):
        # get all indices of stimulus: 
        indices = np.nonzero(stimulus_seq == orientations[i])[0] # get first index of orientation that matches desired orientation
        #print(indices)
        # get only indices of stimuli that occur after time startat:
        indices = indices[indices>=startat_idx]
        #print(indices)
        
        for k in range(N_neurons):        
            
            Ecurrent = []
            Icurrent = []
            for idx in indices:
                # get spike counts of neuron k that happen during stimulus with index idx
                #print(idx)
                #print(np.nonzero((currents.t>idx*input_time)&(currents.t<idx*input_time+input_time)))
                currentEcurrent = currents.IsynE[k][(currents.t>=idx*input_time)&(currents.t<idx*input_time+stim_time)]
                currentIcurrent = currents.IsynI[k][(currents.t>=idx*input_time)&(currents.t<idx*input_time+stim_time)]
                print(len(currentEcurrent))
                Ecurrent.append(currentEcurrent[:600])
                Icurrent.append(currentIcurrent[:600])
                
            # mean current for all occurences of that stimulus
            print('currents')
            print(Ecurrent)
            print('mean')
            print(np.mean(Ecurrent))
            print('max of mean')
            print(np.max(np.mean(Ecurrent,0)))
            print(np.mean(np.max(Ecurrent,0)))

            ampE[k,i] = np.max(np.mean(Ecurrent,0)) # take mean over list elements to get mean current
            ampI[k,i] = np.max(np.mean(Icurrent,0))
            amp2E[k,i] = np.mean(np.max(Ecurrent,0))
            amp2I[k,i] = np.mean(np.max(Icurrent,0))
            amp3E[k,i] = np.mean(np.mean(Ecurrent,0))
            amp3I[k,i] = np.mean(np.mean(Icurrent,0))


            #print(np.max(Ecurrent))
            #print(np.max(Ecurrent,0))            
            #meanampE=np.mean(np.max(Ecurrent,0))
            #meanampI=np.mean(np.max(Icurrent,0))
        
            currentratio[k,i] = ampE[k,i]/(ampI[k,i]+ampE[k,i]) 
            #currentratiomean[k,i] = meanampE/(meanampI+meanampE) 
                
    return currentratio, currentratiomean, ampE, ampI, amp2E, amp2I, amp3E, amp3I # multiply by correct number to get Hz (what is sampling frequency of Stimmonitor? should be 0.1ms)


def get_spike_response(spike_train, no_stimuli, stim_change_time, first=None, last=None):
    N_neurons = len(spike_train)
    
    #get firing rate for neurons at time of each stimulus when it is presented first (between first and first+stim_change_time) 
    #and last (between last-stim_change_time+1 and last+1):
    response = np.zeros((N_neurons,no_stimuli))
    for k in range(N_neurons):
        for i in range(no_stimuli):
            if not first is None:
                response[k,i] = np.sum(((spike_train[k] > first[i]*second) & (spike_train[k] < first[i]*second + stim_change_time)))
            elif not last is None:
                response[k,i] = np.sum(((spike_train[k] < last[i]*second) & (spike_train[k] > last[i]*second - stim_change_time)))

            else:
                raise ValueError('no time given')
    return response


def get_firingrate(spike_train, starttime, endtime):
    N_neurons = len(spike_train)
    firing_rate = np.zeros((N_neurons))
    
    for k in range(N_neurons):
        firing_rate[k] = np.sum(((spike_train[k] > starttime) & (spike_train[k] < endtime)))/(endtime-starttime)
    return firing_rate
    
def get_tuning(spike_train, stimuli_orientation, stimuli_t, no_stimuli):
    N_neurons = len(spike_train)
    first, last = get_stimulus_times(stimuli_orientation, stimuli_t, no_stimuli)
    
    #get firing rate for neurons at time of each stimulus when it is presented first (between first and first+50ms) 
    #and last (between last-50ms and last):
    first_response = np.zeros((N_neurons,no_stimuli))
    last_response = np.zeros((N_neurons,no_stimuli))
    for k in range(N_neurons):
        for i in range(no_stimuli):
            first_response[k,i] = np.sum(((spike_train[k] > first[i]*second) & (spike_train[k] < first[i]*second + 50*ms)))
            last_response[k,i] = np.sum(((spike_train[k] < last[i]*second) & (spike_train[k] > last[i]*second - 50*ms)))

    return first_response, last_response
    

def plot_tuning(first_response, last_response, name = 'None', avg=True,):
    N_neurons = np.shape(first_response)[0]
    if avg:
        # plot average
        plt.plot([0,1,2,3],[mean(first_response[:,0]),mean(first_response[:,1]),mean(first_response[:,2]),mean(first_response[:,3])], 'k')
        plt.plot([0,1,2,3],[mean(last_response[:,0]),mean(last_response[:,1]),mean(last_response[:,2]),mean(last_response[:,3])], 'r')
        plt.xlabel('orientation')
        plt.xlabel('firing rate')
        plt.title(name)

        plt.tight_layout()
        plt.legend(['before','after'])
        plt.show() 
    
    else:        
        # plot all tuning curves sorted according to population
        plt.plot([0,1,2,3],[first_response[:,0],first_response[:,1],first_response[:,2],first_response[:,3]], 'k')
        plt.plot([0,1,2,3],[last_response[:,0],last_response[:,1],last_response[:,2],last_response[:,3]], 'r')
        plt.xlabel('orientation')
        plt.xlabel('firing rate')

        plt.tight_layout()
        plt.show() 


def get_pop_tuning(first_response, last_response):
    meanfirst = np.zeros((np.shape(first_response)[1], np.shape(first_response)[1]))
    meanlast = np.zeros((np.shape(first_response)[1], np.shape(first_response)[1]))
    for i in range(np.shape(first_response)[1]):
        meanfirst[i,:] = np.mean(first_response[i*100:(i*100)+100,:],0)
        meanlast[i,:] = np.mean(last_response[i*100:(i*100)+100,:],0)
    return meanfirst, meanlast

def plot_PYRtuning(first_response, last_response, avg=True):
    NPYR = np.shape(first_response)[0]
    if avg:
        # plot average over each population
        for i in range(0,NPYR,100):    
            plt.subplot(NPYR/200,NPYR/200,(i/100)+1)
            plt.plot(np.arange(4),[mean(first_response[i:i+100,0]),mean(first_response[i:i+100,1]),mean(first_response[i:i+100,2]),mean(first_response[i:i+100,3])], 'k')
            plt.plot(np.arange(4),[mean(last_response[i:i+100,0]),mean(last_response[i:i+100,1]),mean(last_response[i:i+100,2]),mean(last_response[i:i+100,3])], color = cmaps.viridis(0.6))
            plt.xticks(np.arange(0,4),('|','/','--','\\'))
            plt.xlabel('orientation')
            plt.ylabel('firing rate')
            #plt.ylim(0,np.round(np.max(last_response)+1))
            plt.title('population %d'%((i/100)+1))
            plt.tight_layout()
                
    
    else:        
        # plot all tuning curves sorted according to population
        for i in range(0,NPYR,100):    
            plt.subplot(2,2,(i/100)+1)
            plt.plot([0,1,2,3],[first_response[i:i+100,0],first_response[i:i+100,1],first_response[i:i+100,2],first_response[i:i+100,3]], 'k')
            plt.plot([0,1,2,3],[last_response[i:i+100,0],last_response[i:i+100,1],last_response[i:i+100,2],last_response[i:i+100,3]], 'r')
            plt.xlabel('orientation')
            plt.title('population %d'%((i/100)+1))
            plt.tight_layout()
        plt.show() 
        
def get_avg_tuning_curves(last_response, NPYR, no_stimuli):
    tuning_curves = []
    for i in range(no_stimuli):
        tuning_curves.append(mean(last_response[i*100:(i+1)*100,:],0))
    return tuning_curves       

def get_selectivity(tuning_curves):
    select = np.zeros(len(tuning_curves))
    for i in range(len(tuning_curves)):
        # selectivity towards rewarded
        select[i] = tuning_curves[i][0]/np.mean(tuning_curves[i][0]) 

    mean_select = np.mean(select)
    return select, mean_select

def get_selectivity_change(tuning_curves):
    selectivity = np.zeros(len(tuning_curves))
    for i in range(len(tuning_curves)):
        selectivity[i] = tuning_curves[i][0]/tuning_curves[i][i]
    return selectivity

def get_response_towards_rewarded(tuning_curves, prev_tuning_curves):
    response = np.zeros(len(tuning_curves))
    for i in range(len(tuning_curves)):
        response[i] = (tuning_curves[i][0]-prev_tuning_curves[i][0])/prev_tuning_curves[0][0]
    return response

def plot_selectivity_change(selectivity):
    plt.figure()
    plt.plot([1,2,3],[1,1,1],'-')
    plt.plot([1,2,3],selectivity[1:],'.')
    plt.xticks([1,2,3])
    plt.ylim(0,round(max(selectivity)+1))
    plt.xlabel('population')
    plt.title('selectivity towards rewarded')
    plt.tight_layout()
    plt.show()


def plot_selectivity(response, response_before = None, N_pyr = 4):
    plt.figure()
    plt.plot(np.arange(N_pyr),response,'.r')
    if response_before is not None:
        plt.plot(np.arange(N_pyr),response_before,'.k')
    plt.xticks(np.arange(N_pyr))
    #plt.ylim(0,int(max(response)+1))
    plt.xlabel('population')
    plt.title('selectivity towards rewarded')
    plt.tight_layout()
    plt.show()

def plot_weights(neuron_i,neuron_j,weights, title):
    ax = plot_synapses(neuron_i, neuron_j, weights, var_name='synaptic weights',
                   plot_type='scatter', cmap='hot')
    add_background_pattern(ax)
    ax.set_title(title)
    show()
    
def calculate_response_increase(tuning_before, tuning_after, N_pyr):
    response_rel_to_max = np.zeros((N_pyr))
    response_rel_to_max_before = np.zeros((N_pyr))
    
    for k in range(N_pyr):      
        baseline_tuning_after = np.mean(tuning_after[k,(np.arange(len(tuning_after))!=k) & (np.arange(len(tuning_after))!=0)])
        baseline_tuning_before = np.mean(tuning_before[k,(np.arange(len(tuning_after))!=k) & (np.arange(len(tuning_after))!=0)])

        response_rel_to_max[k] = (tuning_after[k,0]-baseline_tuning_after)/(tuning_after[k,k]-baseline_tuning_after)
        response_rel_to_max_before[k] = (tuning_before[k,0]-baseline_tuning_before)/(tuning_before[k,k]-baseline_tuning_before)
        
    diff = response_rel_to_max - response_rel_to_max_before        
    
    
    return response_rel_to_max, diff



if __name__ == "__main__":
    # example of how to use the above, you could do this in an ipython notebook
    reader = ExperimentReader('./my_runs')
    # load all runs at once
    run_no = '30' 
    runs = reader.get_all_experiment_runs()
    # look at the config for run 1
    config = runs[run_no]['config']
    # look at the results for run 1
    #print(runs['1']['info']['output'])
    #print all artifact names for run 1    
    artifact_names1 = reader.get_artifact_file_names(run_no)
    #print("run %s has artifacts:"%run_no)
    #print(artifact_names1)
    # an load some artifacts
    #print("artifact contents:")
    for fname in artifact_names1:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        # print a few lines
        #print(data[:5])
    # alternatively try to load the data directly with the reader
    # the reader tries to be somewhat smart about what files to read
    all_data = reader.try_loading_artifacts(run_no)
    print(all_data.keys())
    
    spike_train = all_data['PYR_spiketrain.pkl']
    #SOMspike_train = all_data['SOM_spiketrain.pkl']
    #VIPspike_train = all_data['VIP_spiketrain.pkl']

    stimuli_orientation = all_data['stimuli_orientation.pkl']
    stimuli_t = all_data['stimuli_t.pkl']
    
    recurrent_weights = all_data['recurrent_weights.pkl']
    recurrent_i = all_data['recurrent_i.pkl']
    recurrent_j = all_data['recurrent_j.pkl']

    PYR2VIP_weights = all_data['PYR2VIP_weights.pkl']
    PYR2VIP_i = all_data['PYR2VIP_i.pkl']
    PYR2VIP_j = all_data['PYR2VIP_j.pkl']


    first, last = get_tuning(spike_train, stimuli_orientation, stimuli_t, no_stimuli=4)
    plot_PYRtuning(first, last)
    
    plot_weights(recurrent_i, recurrent_j, recurrent_weights, 'recurrent connections')
    plot_weights(PYR2VIP_i, PYR2VIP_j, PYR2VIP_weights, 'PYR to VIP connections')

    tuning_curves = get_avg_tuning_curves(last,400,4)
    prev_tuning_curves = get_avg_tuning_curves(first,400,4)
    select, mean_selectivity = get_selectivity(tuning_curves)
    selectivity = get_selectivity_change(tuning_curves)
    response = get_response_towards_rewarded(tuning_curves, prev_tuning_curves)
    plot_selectivity_change(selectivity)
    plot_selectivity(select)
    plot_selectivity(response)
    
    run_nos = np.arange(42,67)
    mean_selectivity = np.zeros(len(run_nos))
    parameter = np.zeros(len(run_nos))
    parameter2 = np.zeros(len(run_nos))
    parameter_name = 'p_PV_PYR'
    for i, run_no in enumerate(run_nos):
        run_no = str(run_no)
        parameter[i] = runs[run_no]['config']['params'][parameter_name]
        artifact_names1 = reader.get_artifact_file_names(run_no)
        all_data = reader.try_loading_artifacts(run_no)
        
        spike_train = all_data['PYR_spiketrain.pkl']
    
        stimuli_orientation = all_data['stimuli_orientation.pkl']
        stimuli_t = all_data['stimuli_t.pkl']        
    
        first, last = get_tuning(spike_train, stimuli_orientation, stimuli_t, no_stimuli=4)
        plot_PYRtuning(first, last)
    
        tuning_curves = get_avg_tuning_curves(last,400,4)
        prev_tuning_curves = get_avg_tuning_curves(first,400,4)
        select, mean_selectivity[i] = get_selectivity(tuning_curves)
        plot_selectivity(select)
    
    #plt.figure()
    #plot(parameter,mean_selectivity,'.')
    ##plt.ylim(0,int(max(response)+1))
    #plt.xlabel('%s'%parameter_name)
    #plt.ylabel('mean selectivity')
    #tight_layout()
    #plt.show()
    
    # you could also load all artifacts at once via
    #loaded_artifacts = reader.try_loading_artifacts_for_all_runs()
    
    # you could now plot the info, e.g. the output or whatever you saved into artifacts

