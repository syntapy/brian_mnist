import brian2 as br
import numpy as np
import pudb

#pudb.set_trace()

a = 0.2
b = 0.7
c = 2.5

vr = -76*br.mV
vt = -15*br.mV

tau1 = 5.00*br.ms
tau2 = 1.25*br.ms

#Ni = br.NeuronGroup(3, '''dv/dt = (vt - vr)/period : volt (unless refractory)
#                                period: second
#                                fire_once: boolean''', \
#                                threshold='v>vt', reset='v=vr',
#                                refractory='fire_once')
#Ni.period = [1, 1, 7] * br.ms
#Ni.fire_once[:] = [True] * 3

indices = np.asarray([0])
times = np.asarray([6]) * br.ms

Ni = br.SpikeGeneratorGroup(3, indices=indices, times=times)
#Nh = br.NeuronGroup(1, model="""dv/dt=(gtot-v)/(10*ms) : 1
#                                  gtot : 1""")
Nh = br.NeuronGroup(1, model="""v = I :1
                                I : 1""")
S = br.Synapses(Ni, Nh,
           model='''tl : second
                    alpha=exp((tl - t)/tau1) - exp((tl - t)/tau2) : 1
                    w : 1
                    I_post = w*alpha : 1 (summed)
                 ''',
           pre='tl+=t - tl')
S.connect('True')
S.w[:, :] = '(80+rand())'
S.tl[:, :] = '0*second'
M = br.StateMonitor(Nh, 'v', record=True)
br.run(20*br.ms)
br.plot(M[0].t, M[0].v)
br.show()
