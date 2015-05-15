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

def ReSuMe(net, mnist, Pc, N_hidden, T, N_h, N_o, v0, u0, I0, ge0, \
        neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters):

    trained = False
    N_hidden_last = len(net[neuron_names[-2]])
    N_out = len(net[neuron_names[-1]])

    N_h = 1
    N_o = 1

    #pudb.set_trace()
    while trained == False:

        dw = np.zeros(len(net[synapse_names[-1]]))

        trained = True
        for number in range(N_hidden_last):

            N_h = init.out(mnist[1][number][0])
            desired_index = number / 2
            print "number = ", number

            lst = range(N_hidden_last)
            rnd.shuffle(lst)

            k = 0
            for i in lst:
                net = snn.Run(net, mnist, number, T, v0, u0, I0, ge0, \
                            neuron_names, synapse_names, state_monitor_names, \
                            spike_monitor_names, parameters)

                #pudb.set_trace()
                indices_l, spikes_l = net[spike_monitor_names[-1]].it
                indices_i, spikes_i = net[spike_monitor_names[-2]].it

                #right_spike_numbers_hidden = init.check_number_spikes(net, 0, \
                #            T, N_h, N_o, v0, u0, I0, ge0, \
                #            neuron_names, spike_monitor_names)

                #right_spike_numbers_out = init.check_number_spikes(net, 1, \
                #            T, N_h, N_o, v0, u0, I0, ge0, \
                #            neuron_names, spike_monitor_names)

                #if right_spike_numbers_hidden == False or right_spike_numbers_out == False:
                #    if right_spike_numbers_hidden == False:
                #        print "ERROR!! WRONG NUMBER OF SPIKES!! Resetting No. Spikes!!!"
                #        init.set_number_spikes(net, 0, T, N_h, N_o, v0, u0, I0, ge0, \
                #                neuron_names, synapse_names, state_monitor_names, spike_monitor_names, \
                #                parameters)

                #    elif right_spike_numbers_out == False:
                #        print "ERROR!! WRONG NUMBER OF SPIKES!! Resetting No. Spikes!!!"
                #        init.set_number_spikes(net, 1, T, N_h, N_o, v0, u0, I0, ge0, \
                #                neuron_names, synapse_names, state_monitor_names, spike_monitor_names, \
                #                parameters)

                #else:
                #sys.exit()
                S_l = init.collect_spikes(indices_l, spikes_l, 4)
                S_i = init.collect_spikes(indices_i, spikes_i, N_hidden[-1])
                S_d = init.out(number) 

                pudb.set_trace()
                P = P_Index(S_l, S_d)
                print "\tP = ", P
                if P >= Pc:
                    k += 1
                    trained = False
                    print "\t\ti = ", i
                    sd = max(0, float(S_d) - float(S_i[i][0]))
                    sl = max(0, float(S_l[0][0]) - float(S_i[i][0]))
                    Wd = WeightChange(sd)
                    Wl = -WeightChange(sl)
                    net.restore()
                    net[synapse_names[3]].w[i, 0] = net[synapse_names[3]].w[i] + Wd + Wl
                    net.store()
                    if k > 8:
                        break
                else:
                    break

    init._save_weights(net, synapse_names, len(synapse_names)-1, len(synapse_names))
    F = open("weights/trained.txt", 'w')
    F.write("True")
    F.close()

def SpikeSlopes(Mv, S_out, d_i=3):
    
    """
        Returns a list of values that indicate the difference in voltage 
        between each spike's starting threshold voltage and the voltage 
        d_i time steps before it

        NOTE: This assumes that the brian equation solver uses a constant time step
        throught computation.
    """

    N = len(S_out.spikes)
    dt = Mv.times[1] - Mv.times[0]
    v_diffs = []
    i_diffs = []

    for i in range(N):
        time = S_out.spikes[i]
        index_a = time / dt
        index_b = index_a - d_i 

        v_diffs.append(Mv.values[index_a] - Mv.values[index_b])
        i_diffs.append(index_a - index_b)

    return v_diffs, dt

def PickWeightIndexA(Sa, S_hidden, S_out):
    pass

def PickWeightIndicesB(Mv, Sb, S_hidden, S_out, d_i=3):

    """
        Depending on the delays of the synapses, and the spike times
        in the hidden layer and output layer, modification of only certain of the weights
        in the hidden to output synapses will have an effect on each output spike

    """

    v_diffs, i_diffs = SpikeSlopes(Mv, S_out, d_i)


"""
def TestNodeRange(T, v0, u0, I0, ge0, neuron_names, synapse_names, state_monitor_names, spike_monitor_names, \
        parameters, number, net)

    n_hidden_last = len(net[neuron_names[2][-1]])
    old_weights = np.empty(n_hidden_last)

    return_val = [-1, -1]

    for i in range(n_hidden_last):
        old_weights[i] = Sb.w[i]
        Sb.w[i] = 0

    j = 0
    Sb.w[0] = 0
    while True:

        snn.Run(T, v0, u0, I0, ge0, neuron_names, synapse_names, state_monitor_names, spike_monitor_names, \
                parameters, number, net)
        #pudb.set_trace()
        spikes_out = S_out.spiketimes[0]
        #spikes_hidden = S_hidden.spiketimes[0]
        n_outspikes = len(spikes_out)
        print "n_outspikes, Sb.w[0] = ", n_outspikes, ", ", Sb.w[0]

        if n_outspikes == 1:
            if return_val[0] == -1:
                #pudb.set_trace()
                return_val[0] = spikes_out[0]# - spikes_hidden[0]
            return_val[1] = spikes_out[0]
        elif n_outspikes > 1:
            #pudb.set_trace()
            break

        Sb.w[0] = Sb.w[0] + 0.001

        #if j % 1 == 0:
        #    snn.Plot(Mv, 0)
        #    
        #j += 1


    for i in range(n_hidden_last):
        Sb.w[i] = old_weights[i]

    return return_val
"""
