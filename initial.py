import os.path as op
import brian2 as br
import numpy as np
import sys
import pudb
import snn

br.prefs.codegen.target = 'cython'

def _neuronindices(N_hidden):
    """
        Nin, Nli, Nlh, Nlo, N_liq, Nhid, Nout
    """
    return 0, 1, 2, 3, 4, 5, 5+N_hidden

def _synapseindices(N_hidden):
    """Si, Sl, Sa, Sb"""
    return 0, 1, 2, 3+N_hidden

def SetNeuronGroups(N_in, N_liquid, N_hidden, N_out, parameters, \
        eqs_hidden_neurons, reset, neuron_names):

    #pudb.set_trace()
    #x = np.array([0, 1, 2])
    #y = np.array([0, 2, 1])*br.msecond
    #input_neurons = br.SpikeGeneratorGroup(N=N_in+1, indices=x, times=y)
    input_neurons = br.NeuronGroup(N_in, '''dv/dt = (vt - vr)/period : volt (unless refractory)
                                        period: second
                                        fire_once: boolean ''', \
                                    threshold='v>vt', reset='v=vr',
                                    refractory='fire_once', \
                                    name=neuron_names[0])

    a = parameters[0]
    b = parameters[1]
    c = parameters[2]
    d = parameters[3]
    tau = parameters[4]
    vt = parameters[5]
    vr = parameters[6]

    if N_liquid != 0:
        liquid_neurons = br.NeuronGroup(N_liquid[-1], model=eqs_hidden_neurons, \
                threshold='v>vt', refractory=2*br.ms, reset=reset, \
                method='euler', name=neuron_names[1][3])

        liquid_in = br.Subgroup(liquid_neurons, 0, N_liquid[0], \
                name=neuron_names[1][0])
        liquid_hidden = br.Subgroup(liquid_neurons, N_liquid[0], N_liquid[-1] - N_liquid[1], \
                name=neuron_names[1][1])
        liquid_out = br.Subgroup(liquid_neurons, N_liquid[-1] - N_liquid[1], N_liquid[-1], \
                name=neuron_names[1][2])

        liquid_in.indices = np.arange(0, N_liquid[0])
        liquid_hidden.indices = np.arange(N_liquid[0], N_liquid[-1] - N_liquid[1])
        liquid_out.indices = np.arange(N_liquid[-1] - N_liquid[1], N_liquid[-1])

        liquid_in.v = 0*br.mV
        liquid_hidden.v = 0*br.mV
        liquid_out.v = 0*br.mV
    else:
        liquid_in = None
        liquid_hidden = None
        liquid_out = None

    hidden_neurons = []
    for i in range(len(N_hidden)):
        hidden_neurons.append(br.NeuronGroup(N_hidden[i], \
            model=eqs_hidden_neurons, threshold='v>vt', refractory=6*br.ms, reset=reset, \
            method='rk4', name=neuron_names[1]))

    output_neurons = br.NeuronGroup(N_out, model=eqs_hidden_neurons,\
        threshold='v>vt', refractory=6*br.ms, reset=reset, method='rk4', name=neuron_names[2])

    if N_liquid != 0:
        neuron_groups = [input_neurons, \
            [liquid_in, \
            liquid_hidden, \
            liquid_out, \
            liquid_neurons], \
            hidden_neurons, \
            output_neurons]
    
    else:
        neuron_groups = [input_neurons, \
            hidden_neurons, \
            output_neurons]

    return neuron_groups

def _initconditions(net, string, v0, u0, I0, ge0):
    net[string].v = v0
    net[string].u = u0
    net[string].I = I0
    net[string].ge = ge0

def _neuroninitconditions(net, neuron_names, v0, u0, I0, ge0):
    N_groups = len(neuron_names)

    for i in range(N_groups):
        if type(neuron_names[i]) == list:
            N = len(neuron_names[i])
            for j in range(N):
                _initconditions(net, neuron_names[i][j], v0, u0, I0, ge0)
        else:
            _initconditions(net, neuron_names[i], v0, u0, I0, ge0)

    return net

def SetSynapseInitialWeights(net, synapse_names, N_hidden):
    #pudb.set_trace()
    net[synapse_names[0]].connect('True')
    net[synapse_names[0]].w[:]='21.8*(1+10.3*(2.0*rand() - 0.3))'
    net[synapse_names[0]].delay='0*ms'
    #N_hidden = str(N_hidden)
    net[synapse_names[1]].connect('True')
    net[synapse_names[1]].w[:,:]='2.13*(1 + 1.3*(2.0*rand() - 0.3))'
    #net[synapse_names[1]].w[:] *= ''
    net[synapse_names[1]].delay='0*ms'

    #pudb.set_trace()
    #for i in range(len(synapse_names[2])):
    #    net[synapse_names[2][i]].connect(True)
    #    net[synapse_names[2][i]].w[:, :]='15.1*(0.5+0.5*rand())'
    #    net[synapse_names[2][i]].delay='(0)*ms'

    """
    non-zero index     non-zero neuron
    0:                 0
    1:                 1
    2:                 2
    3:                 
    4:                 
    5:                 0
    6:                 1
    7:                 2
    8:                 
    """
    #N_synapses = len(net[synapse_names[2][-1]])
    #N_neurons = 5#len(net[neuron_names[3]])

    #for i in range(0, N_synapses, N_synapses / N_neurons):
    #n = len(net[synapse_names[2][-1]].w[:, 0])
    #A = net[synapse_names[2][-1]].w
    #for i in range(n):
    #    net[synapse_names[2][-1]].w[i, 0] = '7.5'
    #pudb.set_trace()

    #net[synapse_names[2][-1]].w[:, :]='15.1*(0.2+0.5*rand())'
    #net[synapse_names[2][-1]].w[9] = 10
    #net[synapse_names[-1]].connect(True)
    #net[synapse_names[-1]].w[:, :]='1.9*(0.5+0.5*rand())'
    #net[synapse_names[-1]].delay='(0)*ms'

    net.store()
    return net

def SetSynapses(neuron_groups, synapse_names):

    #pudb.set_trace()
    #synapse_names = ['Si', 'Sl', [], 'Sb']
    s = 1.0
    N_hidden_layers = len(neuron_groups[1])
    #pudb.set_trace()
    Si = br.Synapses(neuron_groups[0], neuron_groups[1][0], model='w:1', pre='ge+=w*mV', \
            name=synapse_names[0])
    Sl = br.Synapses(neuron_groups[1][-1], neuron_groups[2], model='w:1', pre='ge+=w*mV', \
            name=synapse_names[1])

    #Sa = []
    #Sa.append(br.Synapses(neuron_groups[1][2], neuron_groups[2][0], model='w:1', pre='ge+=w*mV', \
    #        name=synapse_names[2][0]))
    #for i in range(N_hidden_layers - 1):
    #    Sa.append(br.Synapses(neuron_groups[1][i], neuron_groups[1][i+1], model='w:1', \
    #            pre='ge+=w*mV'), name=synapse_names[2][i+1])
    #Sb = br.Synapses(neuron_groups[2][-1], neuron_groups[3], model='w:1', pre='ge+=w*mV', \
    #        name=synapse_names[3])

    synapse_groups = [Si, Sl]

    return synapse_groups

def _neuron_group_index(index_str):

    if index_str == 'input':
        index = 0
    elif index_str == 'hidden':
        index = 1
    elif index_str == 'output':
        index = 2
    else:
        return None

    return index

def StateMonitors(neuron_groups, index_str, index_record=0):

    index = _neuron_group_index(index_str)
    if index_str == 'input':
        M = br.StateMonitor(neuron_groups[index], 'v', record=index_record, \
                    name=(index_str + '_v'))

        return M

    elif index_str == 'hidden':
        Mge = br.StateMonitor(neuron_groups[index][0], 'ge', record=index_record, \
                    name=(index_str + '_ge' + str(index_record)))
        Mv = br.StateMonitor(neuron_groups[index][0], 'v', record=index_record, \
                    name=(index_str + '_v' + str(index_record)))
        Mu = br.StateMonitor(neuron_groups[index][0], 'u', record=index_record, \
                    name=(index_str + '_u' + str(index_record)))
    else:
        Mge = br.StateMonitor(neuron_groups[index], 'ge', record=index_record, \
                    name=(index_str + '_ge' + str(index_record)))
        Mv = br.StateMonitor(neuron_groups[index], 'v', record=index_record, \
                    name=(index_str + '_v' + str(index_record)))
        Mu = br.StateMonitor(neuron_groups[index], 'u', record=index_record, \
                    name=(index_str + '_u' + str(index_record)))

    return Mv, Mu, Mge

def SpikeMonitor(neuron_groups, index_str):
    index_a, index_b, index_aux = _neuron_group_index(index_str)

    if index_b == None and index_aux == None:
        S = br.SpikeMonitor(neuron_groups[index_a], record=0)
    elif index_a == 2:
        S = br.SpikeMonitor(neuron_groups[index_a][index_aux], record=0)
    elif index_b != None:
        S = br.SpikeMonitor(neuron_groups[index_a][index_b], record=0)

    return S

def AllSpikeMonitors(neuron_groups, spike_monitor_names):
    N = len(neuron_groups)

    spike_monitors = []

    spike_monitors.append(br.SpikeMonitor(neuron_groups[0], record=0, name=spike_monitor_names[0]))
    spike_monitors.append(br.SpikeMonitor(neuron_groups[1][0], record=0, name=spike_monitor_names[1]))
    #spike_monitors.append([])
    #for i in range(len(neuron_groups[2])):
    #    spike_monitors[2].append(br.SpikeMonitor(neuron_groups[2][i], \
    #            record=0, name=spike_monitor_names[2][i]))
    spike_monitors.append(br.SpikeMonitor(neuron_groups[2], record=0, name=spike_monitor_names[2]))

    return spike_monitors

def _network(net, group):
    N_groups = len(group)

    if N_groups > 0:
        for i in range(N_groups):
            if type(group[i]) == list:
                N = len(group[i])
                for j in range(N):
                    net.add(group[i][j])
            else:
                net.add(group[i])
    else:
        net.add(group)

    return net

def AddNetwork(neuron_groups, synapse_groups, state_monitors, spike_monitors, parameters):
    a = parameters[0]
    b = parameters[1]
    c = parameters[2]
    d = parameters[3]
    tau = parameters[4]
    vt = parameters[5]
    vr = parameters[6]

    net = br.Network()

    net = _network(net, neuron_groups)
    net = _network(net, synapse_groups)
    net = _network(net, spike_monitors)
    if type(state_monitors) == list or type(state_monitors) == tuple:
        for i in range(len(state_monitors)):
            net = _network(net, state_monitors[i])
    else:
        net = _network(net, state_monitors)

    net.store()
    return net

def SetInitStates(net, N_in, vr, v0, u0, I0, ge0, neuron_names, bench='xor'):

    """
    Sets initial values of the variables, but input states are not set.
    """

    #for number in range(4):
    net = _neuroninitconditions(net, neuron_names[1:], v0, u0, I0, ge0)
    letter = None
    label = 0
    """
    img, label = snn.ReadImg(number=number, bench=bench, letter=letter)
    spikes = snn.GetInSpikes(img, bench=bench)
    net[neuron_names[0]].period = spikes * br.ms
    """
    net[neuron_names[0]].fire_once = [True] * N_in
    net[neuron_names[0]].v = vr
    #net.store(str(number))

    #net.restore('0')
    #print N_in
    net.store()
    #pudb.set_trace()
    #sys.exit()

    return net

def _modify_weights(S, dv):
    n = len(S.w[:])
    for i in range(n):
        weet = S.w[i]
        weet = weet*br.volt + dv*br.mV
        S.w[i] = weet

    return S

def collect_spikes(indices, spikes, N_neurons):
    """
    This takes the indices and spike times from the spike monitor and produces a dictionary
    of the form [[ta1, ta2, ...], [tb1, tb2, ...], ...]
    """

    spikes_hidden = []
    spikes_out = []

    spikes_list = []

    j = 0
    #if len(indices) == 1:
    #    pudb.set_trace()
    arg_sort = br.argsort(indices)
    sorted_indices = br.sort(indices)

    #if len(indices) > 0:
    for i in range(N_neurons):
        spikes_list.append([])

    for i in range(len(sorted_indices)):
        index = arg_sort[i]
        spikes_list[sorted_indices[i]].append(spikes[arg_sort[i]])

    return spikes_list

def _same_num_spikes(indices):
    n = len(indices)

    #pudb.set_trace()
    if n == 0 or n == 1:
        return True

    indices = np.sort(indices)
    q = indices[0]
    count, count_new = -1, 1
    for i in xrange(1, len(indices)):
        if indices[i] == q:
            count_new += 1
        elif count == -1:
            count = count_new
            count = 1
        elif count != count_new:
            return False
        else: count_new = 1

    if count != -1 and count != count_new:
        return False

    return True

def check_number_spikes(net, layer, T, N_h, N_o, v0, u0, I0, ge0, \
        neuron_names, spike_monitor_names):

    """
    Returns True if each hidden neuron is emmitting N_h spikes
    and if each output neuron is emmitting N_o spikes
    """

    N_out_spikes = []
    N_hidden_spikes = []

    # Maybe not good idea to do this on all hidden layers
    N_hidden = len(neuron_names[2])
    N_out = 1

    if layer == 0:
        for i in range(N_hidden):
            hidden_layer = net[neuron_names[2][i]]
            spike_monitor = net[spike_monitor_names[2][i]]
            indices, spikes = spike_monitor.it
            if len(indices) != N_h*len(hidden_layer):
                return False
            if _same_num_spikes(indices) == False:
                return False

    else:
        output_layer = net[neuron_names[3]]
        spike_monitor = net[spike_monitor_names[3]]
        indices, spikes = spike_monitor.it
        if len(indices) != N_o*len(output_layer):
            return False
        if _same_num_spikes(indices) == False:
            return False

    return True

def _modify_neuron_weights(net, neuron_str, synapse_str, neuron_index, dv, N_neurons):
    N_neurons = len(net[neuron_str])
    N_synapses = len(net[synapse_str])
    A = net[synapse_str].w
    n = len(A[:, neuron_index])
    for i in range(n):
        net[synapse_str].w[i, neuron_index] += dv*br.rand()

    return net

def _modify_layer_weights(net, spikes, neuron_str, synapse_str, number, dw_abs, D_spikes):

    modified = False
    N_neurons = len(net[neuron_str])
    if N_neurons == 1:
        index = 0
    else:
        index = number

    if type(D_spikes) == list:
        
        #pudb.set_trace()
        for i in range(index, N_neurons, 4):
            for j in range(N_neurons):
                #pudb.set_trace()
                #print "\t\t\t\tj, spikes[j] = ", j, ", ", spikes[j]
                if len(spikes[j]) > D_spikes[j]:
                    #pudb.set_trace()
                    modified = True
                    net = _modify_neuron_weights(net, neuron_str, synapse_str, j, -dw_abs, N_neurons)
                elif len(spikes[j]) < D_spikes[j]:
                    #pudb.set_trace()
                    modified = True
                    net = _modify_neuron_weights(net, neuron_str, synapse_str, j, dw_abs, N_neurons)

    else:
        for i in range(index, N_neurons, 4):
            for j in range(N_neurons):
                #pudb.set_trace()
                #print "\t\t\t\tj, spikes[j] = ", j, ", ", spikes[j]
                if len(spikes[j]) > D_spikes:
                    #pudb.set_trace()
                    modified = True
                    net = _modify_neuron_weights(net, neuron_str, synapse_str, j, -dw_abs, N_neurons)
                elif len(spikes[j]) < D_spikes:
                    #pudb.set_trace()
                    modified = True
                    net = _modify_neuron_weights(net, neuron_str, synapse_str, j, dw_abs, N_neurons)

    return modified, net

def _basic_training(net, neuron_str, synapse_str, spike_monitor_str, number, dw_abs, D_spikes):

    """
    Modifies the weights leading to each neuron in either the hidden layer or the output layer,
    in order to take it a step closer to having the desired number of spikes
    """

    #pudb.set_trace()
    layer_neurons = net[neuron_str]
    layer_synapses = net[synapse_str]
    spike_monitor = net[spike_monitor_str]
    N_neurons = len(layer_neurons)

    indices, spikes = spike_monitor.it
    #print spikes, "\t", indices
    #pudb.set_trace()
    spikes = collect_spikes(indices, spikes, N_neurons)
    net.restore()
    modified, net = _modify_layer_weights(net, spikes, neuron_str, synapse_str, number, dw_abs, D_spikes)
    net.store()

    return modified, net

def out(label):
    if label == 0:
        N_h = [0, 0, 0, 1]
    elif label == 1:
        N_h = [0, 0, 1, 0]
    elif label == 2:
        N_h = [0, 0, 1, 1]
    elif label == 3:
        N_h = [0, 1, 0, 0]
    elif label == 4:
        N_h = [0, 1, 0, 1]
    elif label == 5:
        N_h = [0, 1, 1, 0]
    elif label == 6:
        N_h = [0, 1, 1, 1]
    elif label == 7:
        N_h = [1, 0, 0, 0]
    elif label == 8:
        N_h = [1, 0, 0, 1]
    elif label == 9:
        N_h = [1, 0, 1, 0]

    return N_h

def out_inverse(S_d):
    #pudb.set_trace()
    if S_d == [0, 0, 0, 1]:
        return 0
    elif S_d == [0, 0, 1, 0]:
        return 1
    elif S_d == [0, 0, 1, 1]:
        return 2
    elif S_d == [0, 1, 0, 0]:
        return 3
    elif S_d == [0, 1, 0, 1]:
        return 4
    elif S_d == [0, 1, 1, 0]:
        return 5
    elif S_d == [0, 1, 1, 1]:
        return 6
    elif S_d == [1, 0, 0, 0]:
        return 7
    elif S_d == [1, 0, 0, 1]:
        return 8
    elif S_d == [1, 0, 1, 0]:
        return 8

def set_number_spikes(net, mnist, layer, T, N_h, N_o, v0, u0, I0, ge0, neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters):

    """
    This sets the number of spikes in the last hidden layer, and in the output layer, to
    N_h and N_o, respectively

    The network should start off with the last hidden layer having small but random weights 
    feeding into it such that the last hidden layer produces no spikes. Then,
    for each neuron in the last hidden layer, the weights feeding into it are gradually increased
    randomly through addiction of small numbers. If the number of spikes is too much, small random
    values are subtracted from the weights leading to it, until the desired number of spikes is
    emitted for every single input value produced in the input neurons.

    One issue is that it may take a long time to do this for more than one input sequence to the
    network as a whole, because the operations done for one input would be somewhat reversing
    the operations done for the other input, hence the likely usefullness of modifcation through
    random values.

    For each input combination that is fed into the network as a whole, it might help to have 
    different vectors which corresond to modification of weights. For instance, for network input
    0, you could modify every 4th neuron, for network input 1 you could modify every forth neuron
    but with an offset of 1, for network input 2 you modify every 4th neuron with an offset of 2,
    and so forth. That might be usefull.
    """

    dw_abs = 0.02
    min_dw_abs = 0.001
    i = 0

    print "layer = ", layer
    if layer == 0:
        dw_abs = 0.5
        #right_dw_abs = True
    else:
        dw_abs = 0.5
        #div = 0
    modified = True
    j = 0
    N = len(mnist[0])

    # Loop until no more modifications are made
    while modified == True:

        modified = False
        print "\tj = ", j
        j += 1
        k = 0

        # Loop over the different input values
        for number in xrange(N):
            #has_desired_spike_number = False
            print "\t\tNumber = ", number, "\t"
            if layer == 0:

                while True:
                    snn.Run(net, mnist, number, T, v0, u0, I0, ge0, \
                            neuron_names, synapse_names, state_monitor_names, \
                            spike_monitor_names, parameters)

                    #print "\t\t\tk = ", k, "\t",

                    k_modified, net = _basic_training(net, \
                            neuron_names[1], synapse_names[0], spike_monitor_names[1], number, dw_abs, N_h)

                    if k_modified == True:
                        modified = True
                    else:
                        break

                    k += 1

            elif layer == 1:

                #pudb.set_trace()
                N_h = out(mnist[1][number][0])

                while True:
                    snn.Run(net, mnist, number, T, v0, u0, I0, ge0, \
                            neuron_names, synapse_names, state_monitor_names, \
                            spike_monitor_names, parameters)

                    #pudb.set_trace()
                    k_modified, net = _basic_training(net, \
                            neuron_names[2], synapse_names[1], spike_monitor_names[2], number, dw_abs, N_h)

                    if k_modified == True:
                        modified = True
                    else:
                        break

                    k += 1
        if layer == 1:
            break

    return net

def _save_single_weight(synapses, file_name_w, file_name_d):

    F = open(file_name_w, 'w')
    G = open(file_name_d, 'w')
    n = len(synapses.w[:])
    for i in range(n):
        F.write(str(synapses.w[i]))
        F.write('\n')
        G.write(str(synapses.delay[:][i] / br.msecond))
        G.write('\n')
    F.close()
    G.close()

def _save_weights(net, synapse_names, a, b):
    for i in range(a, b):
        if type(synapse_names[i]) == list:
            for j in range(len(synapse_names[i])):
                synapse = net[synapse_names[i][j]]
                file_name_w = 'weights/' + synapse_names[i][j] + '_w.txt'
                file_name_d = 'weights/' + synapse_names[i][j] + '_d.txt'
                _save_single_weight(synapse, file_name_w, file_name_d)
        else:
            #pudb.set_trace()
            synapse = net[synapse_names[i]]
            file_name_w = 'weights/' + synapse_names[i] + '_w.txt'
            file_name_d = 'weights/' + synapse_names[i] + '_d.txt'
            _save_single_weight(synapse, file_name_w, file_name_d)

def _save_weights_meta(net, synapse_names, wset=None):
    if wset == 0:
        _save_weights(net, synapse_names, 0, len(synapse_names)-1)
    elif wset == 1:
        _save_weights(net, synapse_names, len(synapse_names) - 1, len(synapse_names))
    else:
        _save_weights(net, synapse_names, 0, len(synapse_names))

def _string_to_weights(string, unit):
    """
    string is a set of floating point numbers or integers separated by newline characters
    """

    #pudb.set_trace()
    n = len(string)
    weights = np.empty(n, dtype=float)
    for i in xrange(n):
        weights[i] = float(string[i][:-1])

    return weights*unit

def _read_weights(synapse_names, a, b):
    weight_list, delay_list = [], []
    for i in range(a, b):
        if type(synapse_names[i]) == list:
            weight_list.append([])
            delay_list.append([])
            for j in range(len(synapse_names[i])):
                file_name_w = 'weights/' + synapse_names[i][j] + '_w.txt'
                file_name_d = 'weights/' + synapse_names[i][j] + '_d.txt'
                F = open(file_name_w, 'r')
                G = open(file_name_d, 'r')
                weight_array = _string_to_weights(F.readlines(), 1)
                delay_array = _string_to_weights(G.readlines(), br.msecond)
                weight_list[i].append(weight_array)
                delay_list[i].append(delay_array)
                F.close()
                G.close()
        else:
            file_name_w = 'weights/' + synapse_names[i] + '_w.txt'
            file_name_d = 'weights/' + synapse_names[i] + '_d.txt'
            F = open(file_name_w, 'r')
            G = open(file_name_d, 'r')
            weight_array = _string_to_weights(F.readlines(), 1)
            delay_array = _string_to_weights(G.readlines(), br.msecond)
            weight_list.append(weight_array)
            delay_list.append(delay_array)
            F.close()
            G.close()

    return br.array(weight_list), br.array(delay_list)

def _save_network_weights(net, synapse_names):
    for i in range(len(synapse_names)):
        if type(synapse_names[i]) == list:
            for j in range(len(synapse_names[i])):
                _save_weights(net[synapse_names[i][j]], 'weights/' + synapse_names[i][j] + '.txt')
        else:
            _save_weights(net[synapse_names[i]], 'weights/' + synapse_names[i] + '.txt')

def _read_network_weights(net, synapse_names):
    for i in xrange(len(synapse_names)):
        if type(synapse_names[i]) == list:
            for j in xrange(len(synapse_names[i])):
                net[synapse_names[i][j]].w[:], net[synapse_names[i][j]].delay[:] = _read_weights(synapse_names[i][j])
        else:
            net[synapse_names[i]].w[:], net[synapse_names[i][j]].delay[:] = _read_weights(synapse_names[i])

    return net

def _number_lines(synapse_name_single):
    with open(synapse_name_single) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def _compatible_dimensions(net, synapse_name_single):
    file_name_w = 'weights/' + synapse_name_single + '_w.txt'
    file_name_d = 'weights/' + synapse_name_single + '_d.txt'
    n_f = _number_lines(file_name_w)
    n_g = _number_lines(file_name_d)
    n_net_w = len(net[synapse_name_single].w[:])
    n_net_d = len(net[synapse_name_single].delay[:])

    return n_f == n_net_w and n_g == n_net_d

def _correct_weights_exist(net, synapse_names, a, b):

    for i in range(a, b):
        if type(synapse_names[i]) == list:
            for j in range(len(synapse_names[i])):
                file_name_w = 'weights/' + synapse_names[i][j] + '_w.txt'
                file_name_d = 'weights/' + synapse_names[i][j] + '_d.txt'
                if op.isfile(file_name_w) == False:
                    return False
                elif op.isfile(file_name_d) == False:
                    return False
                elif _compatible_dimensions(net, synapse_names[i][j]) == False:
                    return False
        else:
            file_name_w = 'weights/' + synapse_names[i] + '_w.txt'
            file_name_d = 'weights/' + synapse_names[i] + '_d.txt'
            if op.isfile(file_name_w) == False:
                return False
            elif op.isfile(file_name_d) == False:
                return False
            elif _compatible_dimensions(net, synapse_names[i]) == False:
                return False

    return True

def _readweights(net, synapse_names, a, b):

    #pudb.set_trace()
    weights, delays = _read_weights(synapse_names, a, b)
    for i in range(a, b):
        if type(synapse_names[i]) == list:
            for j in range(len(synapse_names[i])):
                net[synapse_names[i][j]].w[:] = weights[i][j]
                net[synapse_names[i][j]].delay[:] = delays[i][j]
        else:
            net[synapse_names[i]].w[:] = weights[i]
            net[synapse_names[i]].delay[:] = delays[i]

    return net

def _trained():
    if op.isfile("weights/trained.txt") == True:
        return True
    return False

def SetWeights(net, mnist, N_liquid, N_hidden, T, N_h, N_o, v0, u0, I0, ge0, \
         neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters):

    trained = False
    pudb.set_trace()
    if False:
        if _correct_weights_exist(net, synapse_names, 0, len(synapse_names)):
            net = _readweights(net, synapse_names, 0, len(synapse_names))
            if _trained():
                trained = True
            net.store()
        elif _correct_weights_exist(net, synapse_names, 0, len(synapse_names)-1):
            net = _readweights(net, synapse_names, 0, len(synapse_names)-1)
            net.store()
            net = set_number_spikes(net, mnist, 1, T, N_h, N_o, v0, u0, I0, ge0, \
                    neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters)
            net.store()
            _save_weights(net, synapse_names, len(synapse_names)-1, len(synapse_names))
        else:
            net = set_number_spikes(net, mnist, 0, T, N_h, N_o, v0, u0, I0, ge0, \
                    neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters)

            net.store()
            _save_weights(net, synapse_names, 0, len(synapse_names)-1)

            net = set_number_spikes(net, mnist, 1, T, N_h, N_o, v0, u0, I0, ge0, \
                    neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters)

            net.store()
            _save_weights(net, synapse_names, len(synapse_names)-1, len(synapse_names))

    #pudb.set_trace()
    return net, trained

def ReadTimes(file_name):
    F = open(file_name, 'r') 
    strings = F.readlines()
    desired_times = [-1, -1]
    desired_times[0] = float(strings[0][:-1])*br.second
    desired_times[1] = float(strings[1][:-1])*br.second
    F.close()

    return desired_times

def GetSpikes(net, T, v0, u0, I0, ge0, neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters, number=5):

    indices, spikes = net[spike_monitor_names[3]].it
    spikes_out = collect_spikes(indices, spikes, 1)
    n_outspikes = len(spikes_out[0])

    return n_outspikes, spikes_out

def GetWeightRange(net, T, num_spikes, v0, u0, I0, ge0, neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters, number=5):

    #pudb.set_trace()
    net.restore()
    n_hidden_last = len(net[neuron_names[2][-1]])
    old_weights = np.empty(n_hidden_last)

    extreme_spikes = [-1, -1]

    w_last = 1
    net[synapse_names[3]].w[:, 0] = np.zeros(n_hidden_last, dtype=float)
    net[synapse_names[3]].w[0, 0] = w_last
    net.store('5')

    j = 0
    print "\tDetermining weight range:"
    while True:

        net = snn.Run(net, 2*T, v0, u0, I0, ge0, neuron_names, synapse_names, \
                state_monitor_names, spike_monitor_names, \
                parameters, number=number)

        #pudb.set_trace()
        n_outspikes, spikes_out = GetSpikes(net, T, v0, u0, I0, ge0, neuron_names, synapse_names, \
                    state_monitor_names, spike_monitor_names, \
                    parameters, number=number)

        #pudb.set_trace()
        print "\t\tj, w, n_outspikes = ", j, ", ", net[synapse_names[3]].w[0, 0][0], ", ", n_outspikes
        print "\t\t\tspikes_out",  spikes_out[0]
        if n_outspikes < num_spikes:
            w_last = net[synapse_names[3]].w[0, 0]
            net.restore('5')
            w_new = 2*w_last
            net[synapse_names[3]].w[0, 0] = w_new
            net.store('5')
        elif n_outspikes == num_spikes:
            #w_last = 
            net.restore('5')
            w_new = 1.2*net[synapse_names[3]].w[0, 0]
            net[synapse_names[3]].w[0, 0] = w_new
            net.store('5')
        elif n_outspikes > num_spikes:
            break

        j += 1

    #w_low, w_high = w_last, net[synapse_names[3]].w[0, 0]
    net.restore()

    return net, w_last, w_new

def BinSearchWeights(net, T, num_spikes, w_low, w_high, v0, u0, I0, ge0, neuron_names, synapse_names, \
        state_monitor_names, spike_monitor_names, parameters, end='low'):

    j = 0
    print "\tHoming in on ", end, " bound:"
    while True:

        net.restore('5')
        net[synapse_names[3]].w[0, 0] = (w_low + w_high) / 2
        net.store('5')
        snn.Run(net, T, v0, u0, I0, ge0, neuron_names, synapse_names, \
                state_monitor_names, spike_monitor_names, \
                parameters, 5)
        j += 1
        n_outspikes, spikes_out = GetSpikes(net, T, v0, u0, I0, ge0, neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters, number=5)
        net.restore('5')

        print "\t\tw_low, w_high, n_outspikes = ", w_low, ", ", w_high, ", ", n_outspikes
        print "\t\t\tspikes_out",  spikes_out[0]
        if end == 'low':
            if n_outspikes < num_spikes:
                w_low = (w_low + w_high) / 2
            elif n_outspikes >= num_spikes:
                if n_outspikes == num_spikes and abs(w_high - w_low) < 0.01:
                    break
                w_high = (w_low + w_high) / 2
        else:
            if n_outspikes <= num_spikes:
                if n_outspikes == num_spikes and abs(w_high - w_low) < 0.01:
                    break
                w_low = (w_low + w_high) / 2
            elif n_outspikes > num_spikes:
                w_high = (w_low + w_high) / 2

    return spikes_out[0]

def TestNodeRange(net, T, num_spikes, v0, u0, I0, ge0, neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters):

    #net.restore('4')
    net, w_low, w_high = GetWeightRange(net, T, num_spikes, v0, u0, I0, ge0, neuron_names, synapse_names, \
                        state_monitor_names, spike_monitor_names, \
                        parameters)

    spike_low = BinSearchWeights(net, T, num_spikes, w_low, w_high, v0, u0, I0, ge0, neuron_names, synapse_names, \
        state_monitor_names, spike_monitor_names, parameters, end='low')

    spike_high = BinSearchWeights(net, T, num_spikes, w_low, w_high, v0, u0, I0, ge0, neuron_names, synapse_names, \
        state_monitor_names, spike_monitor_names, parameters, end='high')

    #pudb.set_trace()
    if num_spikes == 1:
        extreme_spikes = [-1, -1]
        extreme_spikes[0] = spike_low[0]
        extreme_spikes[1] = spike_high[0]

    net.restore()
    return extreme_spikes

def OutputTimeRange(net, T, N_h, N_o, v0, u0, I0, ge0, neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters):

    if op.isfile("weights/times.txt"):
        print "Reading desired times"
        desired_times = ReadTimes("weights/times.txt")
    else:
        print "Calculating desired times:"
        desired_times = [-1, -1]
        extreme_spikes = TestNodeRange(net, T, N_o, v0, u0, I0, ge0, \
                neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters)
        diff = extreme_spikes[0] - extreme_spikes[1]
        diff_r = diff / 5

        extreme_spikes[0] = extreme_spikes[0] - diff_r
        extreme_spikes[1] = extreme_spikes[1] + diff_r

        desired_times[0] = extreme_spikes[0]*br.second
        desired_times[1] = extreme_spikes[1]*br.second

        F = open("weights/times.txt", 'w')
        F.write(str(float(desired_times[0])))
        F.write("\n")
        F.write(str(float(desired_times[1])))
        F.write("\n")
        F.close()

        print "\tPrinted to file: ", desired_times

    return desired_times
