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

c1 = scipy.io.loadmat('data1-600.mat')['c1'][0]
#pudb.set_trace()
N = len(c1)
features = np.empty(N, dtype=object)

for i in xrange(N):
    features[i] = rflatten(c1[i])

labels = scipy.io.loadmat('labels.mat')['labels_body']
mnist = [features, labels]
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

N_in = 2
N_liquid = 0#[4, 5, 12] # Total, liquid in, liquid out
#CP_liquid = 0.7
N_hidden = [14]
N_out = 4

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
u0 = (25*(-5*A*B + A**2 * B**2)) * br.mV
v0 = (25*(-5 + A**2 * B**2)) * br.mV
I0 = 0*br.mV / br.ms
ge0 = 0*br.mV
u0 = -8.588384*br.mV
v0 = vr

img = np.empty(img_dims)

count = 0
g = 2

T = 30
N_h = 1
N_o = 1

# DEFINE OBJECTS
neuron_names = ['input', 'hidden', 'out']
synapse_names = ['Si', 'Sl']
state_monitor_names = ['out_ge', 'out_v', 'out_u']
spike_monitor_names = ['sm_in', 'sm_h', 'sm_out']

#pudb.set_trace()
N_in = len(mnist[0][0])#(XX[0] * XX[1] * XX[2]) + (YY[0] * YY[1] * YY[2])
print N_in

#N = 1

neuron_groups = init.SetNeuronGroups(N_in, N_liquid, N_hidden, N_out, \
            parameters, eqs_hidden_neurons, reset, neuron_names)
synapse_groups = init.SetSynapses(neuron_groups, synapse_names)

state_monitor_in = init.StateMonitors(neuron_groups, 'input', index_record=1)
state_monitor_hidden = init.StateMonitors(neuron_groups, 'hidden', index_record=0)
state_monitor_out = init.StateMonitors(neuron_groups, 'output', index_record=0) 

spike_monitors = init.AllSpikeMonitors(neuron_groups, spike_monitor_names)
state_monitors = [state_monitor_in, state_monitor_hidden, state_monitor_out]

net = init.AddNetwork(neuron_groups, synapse_groups, state_monitors, spike_monitors, parameters)
net = init.SetSynapseInitialWeights(net, synapse_names)
net = init.SetInitStates(net, N_in, vr, v0, u0, I0, ge0, neuron_names)
#pudb.set_trace()
net, trained = init.SetWeights(net, mnist, N_liquid, N_hidden, T, N_h, N_o, v0, u0, I0, ge0, \
                neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters)
#desired_times = init.OutputTimeRange(net, T, N_h, N_o, v0, u0, I0, ge0, \
#                neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters)

#if trained == False:
#pudb.set_trace()
net = train.ReSuMe(net, mnist, Pc, N_hidden, T, N_h, N_o, v0, u0, I0, ge0, \
                neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters)

#outputs = [-1, -1, -1, -1]
#
#for number in range(4):
#    net = snn.Run(net, T, v0, u0, I0, ge0, neuron_names, \
#            synapse_names, state_monitor_names, spike_monitor_names, parameters, number)
#
#    indices_l, spikes_l = net[spike_monitor_names[-1]].it
#    outputs[number] = spikes_l[0]
#    print "number, out, desired_out: ", number, ", ", spikes_l[0], ", ", desired_times[number / 2]
    #snn.Plot(state_monitor_out, number)

    #snn.Plot(state_monitor_a, number)
    #snn.Plot(state_monitor_b, number)
    #snn.Plot(state_monitor_c, number)
    #snn.Plot(state_monitor_d, number)
    #snn.Plot(state_monitor_e, number)
    #pudb.set_trace()
    #tested = snn.CheckNumSpikes(T, 1, 1, v0, u0, I0, ge0, neuron_names, spike_monitor_names, net)
#
#snn.SetNumSpikes(T, N_h, N_o, v0, u0, I0, ge0, number, net)

# LIQUID STATE MACHINE

#pudb.set_trace()
#for i in range(len(hidden_neurons)):
#    S_hidden.append(br.SpikeMonitor(hidden_neurons[i], record=True))
#
#S_out = br.SpikeMonitor(output_neurons, record=True)

#objects.append(input_neurons)
#objects.append(output_neurons)
#for i in range(len(hidden_neurons)):
#    objects.append(hidden_neurons[i])
#
#objects.append(S_in)
#objects.append(S_out)
#for i in range(len(hidden_neurons)):
#    objects.append(S_hidden[i])
#
#for i in range(len(N_hidden)):
#    objects.append(Sa[i])
#
#objects.append(Sb)
#
#objects.append(M)
#objects.append(Mv)
#objects.append(Mu)

#net = br.Network(objects)
#pudb.set_trace()

'''         TRAINING        '''
#Net = br.Network(objects)
#OUT = open('weights.txt', 'a')

#number = 3
#N_o = 1
#N_h = 1
#for i in range(10):
#    for number in range(3, -1, -1):
#        snn.SetNumSpikes(T, N_h, N_o, v0, u0, bench, number, \
#            input_neurons, liquid_neurons, hidden_neurons, output_neurons, \
#            Si, Sl, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=False, letter=None)
#        print "\tDone! for number = ", number


"""
print "======================================================================"
print "\t\t\tSetting number of spikes"
print "======================================================================"

pudb.set_trace()
if op.isfile(weight_file):
    #pudb.set_trace()
    Si, Sl, Sa, Sb = snn.ReadWeights(Si, Sl, Sa, Sb, weight_file)

else:
    snn.SaveWeights(Si, Sl, Sa, Sb, "weights.txt")

#pudb.set_trace()
#Sa[0].w[:] = '0*br.mV'
snn.Run(T, v0, u0, bench, 0, input_neurons, hidden_neurons, output_neurons, Si, Sl, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out)
#pudb.set_trace()
#snn.Plot(N, Nu, Nv, 1)
#snn.Plot(M, Mu, Mv, 1)

print "======================================================================"
print "\t\t\tTraining with ReSuMe"
print "======================================================================"

if bench == 'xor':
    if op.isfile("times.txt"):
        desired_times = train.ReadTimes("times.txt")
    else:

        desired_times = [-1, -1]
        extreme_spikes = train.TestNodeRange(T, N, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sl, Sb, M, Mv, Mu, S_in, S_hidden, S_out)
        diff = extreme_spikes[1] + extreme_spikes[0]
        diff_r = diff / 10

        extreme_spikes[0] = extreme_spikes[0] + diff_r
        extreme_spikes[1] = extreme_spikes[1] + diff_r

        desired_times[0] = extreme_spikes[0]*br.second
        desired_times[1] = extreme_spikes[1]*br.second

        f = open("times.txt", 'w')
        f.write(str(float(desired_times[0])))
        f.write("\n")
        f.write(str(float(desired_times[1])))
        f.write("\n")

else:
    pudb.set_trace()

for number in range(4):
    print "\tTRAINING: number = ", number
    train.ReSuMe(desired_times, Pc, T, N, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sl, Sb, M, Mv, Mu, S_in, S_hidden, S_out)

print "======================================================================"
print "\t\t\tTesting"
print "======================================================================"

#pudb.set_trace()
for number in range(4):
    snn.Run(T, v0, u0, bench, number, \
            input_neurons, hidden_neurons, output_neurons, \
            Sa, Sl, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=True, letter=None)

    if number < 2:
        desired = desired_times[0]
    else:
        desired = desired_times[1]

    print "Number, Desired, Actual = ", number, ", ", desired, ", ", S_out.spiketimes[0]
"""
