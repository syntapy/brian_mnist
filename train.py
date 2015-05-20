import brian2 as br
import numpy as np
import math as ma
import random as rnd
import initial as init
import pudb
import snn
import sys

br.prefs.codegen.target = 'cython'

def DesiredOut(label, bench):
    return_val = None

    if bench == 'xor':
        if label == 1:
            return_val = 1*br.ms
        else:
            return_val = 7*br.ms

    return return_val

def WeightChange(s):
    A = 0.5*10**1
    tau = 0.5*br.ms
    return A*ma.exp(-s / tau)

def L(t):
    tau = 5.0*br.ms
    if t > 0:
        return ma.exp(float(-t / tau))
    else:
        return 0

def P_Index(S_l, S_d):
    return_val = 0

    return_val += abs(L(S_d) - L(S_l[0][0]*br.second))

    return return_val

def ReadTimes(filename):
    f = open(filename, 'r')

    #pudb.set_trace()
    lines = f.readlines()
    f.close()

    desired_times = [-1] * len(lines)

    for i in range(len(lines)):
        tmp = float(lines[i][:-1])
        desired_times[i] = tmp * br.second

    return desired_times

def _resume_step(index, ta, tb, w):
    A = 15
    tau = 1.7
    #pudb.set_trace()
    array = tb - ta
    max_indices = np.greater_equal(array, 0)
    max_indices = max_indices.astype(int, copy=False)
    d = array*max_indices
    #a = 0.2

    return A*np.exp(-d / tau)

#def _resume_step_out(index, ta, tb, w):
#    A = 0.7
#    tau = 0.5
#    pudb.set_trace()
#    array = tb - ta
#    max_indices = np.greater_equal(array, 0)
#    max_indices = max_indices.astype(int, copy=False)
#    d = array*max_indices
#    a = 0.2
#
#    return a + A*np.exp(-d / tau)

def _set_out_spike(net, index, S_i, l, d):
    """
        Returns the change in weight for a particular synaptic
        connection between learning neurons and output neurons.
        as computed by ReSuMe-style learning rule.

        However, it is modified from ReSuMe to get neurons
        to spike certain number of times (either 0 or 1) 
        as oposed to certain spike times.

        ToDo: Make this more efficient using numpy
        array handling etc...
    """
    if len(l) != d:
        x, y = 5, 9
        #pudb.set_trace()
        w = net['Sl'].w[:]
        if d == 1:
            #pudb.set_trace()
            dn = 7
            a = _resume_step(index, S_i, x, w)
            b = _resume_step(index, S_i, dn, w)
        elif d == 0:
            dn = l[0]/br.ms
            a = _resume_step(index, S_i, y, w)
            b = _resume_step(index, S_i, dn, w)
        return_val = a - b
        #pudb.set_trace()
        if min(return_val) > 0:
            print "\t\t\t\tGREATER"
            return return_val - min(return_val)
        elif max(return_val) < 0:
            print "\t\t\t\tLESS"
            return return_val + max(return_val)
        else:
            print "\t\t\t\tNONE"
            return return_val

    return 0

def _netoutput(net, spike_monitor_names, N_hidden):
    indices_l, spikes_l = net[spike_monitor_names[-1]].it
    indices_i, spikes_i = net[spike_monitor_names[-2]].it

    S_l = init.collect_spikes(indices_l, spikes_l, 1)
    S_i = init.collect_spikes(indices_i, spikes_i, N_hidden[-1])

    #for i in range(len(S_l)):
    #    if S_l[i] == []:
    #        S_l[i].append(100*br.ms)
    #for i in range(len(S_i)):
    #    if S_i[i] == []:
    #        S_i[i].append(100*br.ms)

    return S_l, S_i

def Compare(S_l, S_d):
    if len(S_l) != len(S_d):
        print "ERROR: Mismatch in tuple length!"
        sys.exit()
    for i in range(len(S_l)):
        if len(S_l[i]) != S_d[i]:
            return False
    return True

def ReSuMe(net, mnist, start, end, Pc, N_hidden, T, N_h, N_o, v0, u0, I0, ge0, neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters):

    #pudb.set_trace()
    trained = False
    N = len(mnist[0])
    N_hidden_last = len(net[neuron_names[-2]])
    N_out = len(net[neuron_names[-1]])

    N_h = 1
    N_o = 1

    #for number in range(start, end):
    number = start - 1
    count = 0
    while count < end - start:

        #print "number = ", number
        dw = np.zeros(len(net[synapse_names[-1]]))

        number += 1

        label = mnist[1][number]
        if label[0] == 0 or label[0] == 1:
            count += 1
            print "number = ", number, "count = ", count
            k = 0
            while True:

                k += 1
                print "\tstep = ", k
                N_h = init.out(mnist[1][number][0])
                desired_index = number / 2

                lst = range(N_hidden_last)
                rnd.shuffle(lst)

                #pudb.set_trace()
                net = snn.Run(net, mnist, number, T, v0, u0, I0, ge0, \
                            neuron_names, synapse_names, state_monitor_names, \
                            spike_monitor_names, parameters)

                #pudb.set_trace()
                S_l, S_i = _netoutput(net, spike_monitor_names, N_hidden)
                S_d = init.out(label)
                #P = P_Index(S_l, S_d)

                print "\t\tS_l = ", S_l
                print "\t\tS_d = ", S_d
                print "\t\tS_i = ", S_i
                #sys.exit()
                #pudb.set_trace()
                #t_min, t_max = min(S_i)[0], max(S_i)[0]

                modified = False
                w = net[synapse_names[-1]].w[:]
                #for j in range(N_out):
                j = 0
                #print "\t\ti = ", j
                #pudb.set_trace()
                t_in_tmp = np.copy(S_i / br.ms)
                t_in = t_in_tmp.flatten()
                #pudb.set_trace()
                #if number == 3 and j == 2:
                #pudb.set_trace()
                dw = _set_out_spike(net, j, t_in, S_l[j], S_d[j])
                #print "\t\t\tj, dw = ", j, ", ", dw
                if type(dw) == np.ndarray:
                    print "\t\t\tdw = ", dw
                    modified = True
                    #w_tmp = w[j::4]
                    w += dw
                #print "\t\t\tDw = ", w - net[synapse_names[-1]].w[:]
                #print "\t\t\tw = ", w
                net.restore()
                net[synapse_names[-1]].w[:] = w
                net.store()
                if modified == False:
                    break

    init._save_weights(net, synapse_names, 0, len(synapse_names))
    F = open("weights/trained.txt", 'w')
    F.write("True")
    F.close()

    return net

def Test(net, mnist, start, end, N_hidden, T, v0, u0, I0, ge0, \
        neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters):

    hit, miss = 0, 0
    hit_ind, miss_ind = np.zeros(10, dtype=int), np.zeros(10, dtype=int)

    print "Testing"
    #for number in range(start, end):
    number = start - 1
    count = 0
    while count < end - start:
        number += 1
        #pudb.set_trace()
        label = mnist[1][number]
        if label[0] == 0 or label[0] == 1:
            count += 1
            print "\tlabel = ", label,
            print "\tnumber = ", number
            net = snn.Run(net, mnist, number, T, v0, u0, I0, ge0, \
                        neuron_names, synapse_names, state_monitor_names, \
                        spike_monitor_names, parameters)
            S_l, S_i = _netoutput(net, spike_monitor_names, N_hidden)
            S_d = init.out(label)
            print "\t\tS_l = ", S_l
            print "\t\tS_d = ", S_d
            print "\t\tS_i = ", S_i
            index = init.out_inverse(S_d)
            result = Compare(S_l, S_d)
            if result == True:
                hit_ind[index] += 1
                hit += 1
            else:
                miss_ind[index] += 1
                miss += 1

    return hit, miss, hit_ind, miss_ind
