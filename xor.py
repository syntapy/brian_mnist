from operator import itemgetter
import scipy.io
import os.path as op
import numpy as np
import brian2 as br
import time
import pudb
import sys
import snn 
import train
import initial as init
#import cProfile
import array as arrr
#pudb.set_trace()

br.prefs.codegen.target = 'cython'

"""
    This script simulates a basic set of feedforward spiking neurons (Izhikevich model).

    So far the ReSuMe algorithm has been implemented (it seems).

    TO DO:
        Clean it up so that it is easier to tweak parameters
        X Test the firing time range of a neuron
        X Add multiple layers
        X Tweak parameters to make it solve the XOR problem efficiently
        X Work this into a script which takes arguments that denote whether it should train the weights and save them to a text file, or read the weights from a text file and just run it.
        Make SetNumSpikes(args...) more efficient, and have the weights made to no be on the border of a change in number of spikes
        Make SetNumSpikes(args...) better at fine tuning the weights, esp. when there are very large numbers of hidden layer neurons
"""

def trace_calls(frame, event, arg):
    if event != 'call':
        return
    co = frame.f_code
    func_name = co.co_name
    if func_name == 'write':
        # Ignore write() calls from print statements
        return
    func_line_no = frame.f_lineno
    func_filename = co.co_filename
    caller = frame.f_back
    caller_line_no = caller.f_lineno
    caller_filename = caller.f_code.co_filename
    print 'Call to %s on line %s of %s from line %s of %s' % \
        (func_name, func_line_no, func_filename,
         caller_line_no, caller_filename)
    return

def rflatten(A):
    if A.dtype == 'O':
        dim = np.shape(A)
        n = len(dim)
        ad = np.zeros(n)
        i = 0
        tmp = []
        for a in A:
            tmp.append(rflatten(a))
        return_val = np.concatenate(tmp)
    else:
        return_val = A.flatten()

    return return_val

c1_train = scipy.io.loadmat('/home/sami/Desktop/mnist-train/train-1.mat')['c1a'][0]
c1_test = scipy.io.loadmat('/home/sami/Desktop/mnist-train/test-1.mat')['c1b'][0]
#pudb.set_trace()
print c1_train[0]
#sys.exit()
#pudb.set_trace()
N_train = len(c1_train)
N_test = len(c1_test)
train_features = np.empty(N_train, dtype=object)
test_features = np.empty(N_test, dtype=object)

for i in xrange(N_train):
    train_features[i] = rflatten(c1_train[i])
for i in xrange(N_test):
    test_features[i] = rflatten(c1_test[i])

train_labels = scipy.io.loadmat('/home/sami/Desktop/mnist-train/train-label.mat')['train_labels_body']
test_labels = scipy.io.loadmat('/home/sami/Desktop/mnist-train/test-label.mat')['test_labels_body']
print train_labels[0][0]
train_mnist = [train_features, train_labels]
test_mnist = [test_features, test_labels]
#c1 = None
#sys.exit()
#sys.settrace(trace_calls)
objects = []
N = 1

vt = -15 * br.mV
vr = -74 * br.mV
vr = -76.151418 * br.mV
A=0.02
B=0.2
C=-65.0
D=6.0
tau=2.0
bench='xor'
levels=4

#N_in = 2
N_liquid = 0#[4, 5, 12] # Total, liquid in, liquid out
#CP_liquid = 0.7
N_hidden = [12]
N_out = 1

#file_array = ["Si", "Sl", "Sa", "Sb"]
#synapes_array = []
Pc = 0.013

'''     0 - Calculates number of filters    '''
levels = 1
n_inputs = 1
img_dims = 1

#XX = np.shape(c1[0][0])
#YY = np.shape(c1[0][1])
#N_in = 1

a = A / br.ms
b = B
c = C*br.mV
d = D*br.mV
tau = tau*br.ms
bench = bench

parameters = [a, b, c, d, tau, vt, vr]

eqs_hidden_neurons = '''
    dv/dt=((0.04/mV)*v**2+(5)*v+140*mV-u+ge)/ms+I           : volt
    du/dt=a*(b*v-u)                                         : volt
    dge/dt=-ge/tau                                          : volt
    I                                                       : volt / second
'''

"""     USING MATHEMATICA
    u = 25. (-5 a b + A B)
    v = 25. (-5. + a b)
"""

reset = '''
    v = c
    u += d
'''

#pudb.set_trace()
u0 = ((25*(-5*A*B + A**2 * B**2)) + 0) * br.mV
v0 = ((25*(-5 + A**2 * B**2)) + 40) * br.mV
I0 = 0*br.mV / br.ms
ge0 = 0*br.mV
u0 = (-8.588384 - 0*12)*br.mV
v0 = vr + 0*10*br.mV

img = np.empty(img_dims)

count = 0
g = 2

T = 10.0
N_h = 1
N_o = 1

# DEFINE OBJECTS
neuron_names = ['input', 'hidden', 'out']
synapse_names = ['Si', 'Sl']
state_monitor_names = ['out_ge', 'out_v', 'out_u']
spike_monitor_names = ['sm_in', 'sm_h', 'sm_out']

#pudb.set_trace()
N_in = len(train_mnist[0][0])#(XX[0] * XX[1] * XX[2]) + (YY[0] * YY[1] * YY[2])
print N_in

neuron_groups = init.SetNeuronGroups(N_in, N_liquid, N_hidden, N_out, \
            parameters, eqs_hidden_neurons, reset, neuron_names)
synapse_groups = init.SetSynapses(neuron_groups, synapse_names)

state_monitor_in = init.StateMonitors(neuron_groups, 'input', index_record=1)
state_monitor_hidden = init.StateMonitors(neuron_groups, 'hidden', index_record=0)
state_monitor_out = init.StateMonitors(neuron_groups, 'output', index_record=0) 

spike_monitors = init.AllSpikeMonitors(neuron_groups, spike_monitor_names)
state_monitors = [state_monitor_in, state_monitor_hidden, state_monitor_out]

net = init.AddNetwork(neuron_groups, synapse_groups, state_monitors, spike_monitors, parameters)
net = init.SetSynapseInitialWeights(net, synapse_names, N_in, N_hidden)
net = init.SetInitStates(net, N_in, vr, v0, u0, I0, ge0, neuron_names)
net, trained = init.SetWeights(net, N_liquid, N_hidden, T, N_h, N_o, v0, u0, I0, ge0, \
                    neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters)

F = open("train_results.txt", 'a')
F.write("\n")
F.write("=====================================\n\n")
F.write(str(N_hidden[0]) + " hidden neuron\n")
#F.write("\t" + "Training on images " + str(start) + " to " str(end) + "\n")

start, end = 0, 8#0#00#0
start_time = time.time()
if trained == False:
    net = train.ReSuMe(net, train_mnist, start, end, Pc, N_hidden, T, N_h, N_o, v0, u0, I0, ge0, \
                neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters)

elapsed_time = time.time() - start_time

F.write("\t" + str(elapsed_time) + " to train on mnist images " + str(start) + " to " + str(end) + "\n")

start_time = time.time()
start, end = 0, 8#0#00#0#0
hit, miss, hit_r, miss_r = train.Test(net, train_mnist, start, end, N_hidden, T, v0, u0, I0, ge0, \
                neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters )
elapsed_time = time.time() - start_time

F.write("\t" + str(elapsed_time) + " to test on mnist images " + str(start) + " to " + str(end) + "\n")
F.write("\t" + str(hit) + " hits, " + str(miss) + " misses during testing\n")

F.write("\t" + "Hit array: " + np.array_str(hit_r) + "\n")
F.write("\t" + "Miss array: " + np.array_str(miss_r) + "\n")
F.close()
