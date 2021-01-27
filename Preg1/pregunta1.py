from brian2 import *
from brian2 import StateMonitor
import numpy as np
import matplotlib.pyplot as plt
start_scope()
%matplotlib inline
Temp = np.array([6, 7.2, 9, 11.5, 22.5, 33])
C_m = 1#ufarad*cm**-2
Temp_o = np.ones(len(Temp))*25
V_na = 50
V_sd = 50
V_k = -90
V_sr= -90
V_l = -60
tau = 1.0*ms
tau_k = 2.0#*ms
tau_sr = 20#*ms
tau_sd = 10#*ms
g_na = 1.5#*msiemens*cm**-2
g_sd = 0.25#*msiemens*cm**-2
g_l = 0.1#*msiemens*cm**-2
g_k = 2.0#*msiemens*cm**-2
g_sr = 0.4#*siemens*cm**-2
p_o = exp((Temp-Temp_o)*np.log(1.3)/10)
#print(p)
phi_o = exp((Temp-Temp_o)*np.log(3.0)/10)
#print(phi)
alpha = 0.012#*cm**2/amp
beta = 0.17

eqs ='''
dV/dt = -(p*g_na*a_na*(V - V_na) + p*g_k*a_k*(V - V_k) +
            p*g_sd*a_sd*(V - V_sd) + p*g_sr*a_sr*(V - V_sr) +
            g_l*(V - V_l))/(C_m*tau) : 1
da_k/dt = -phi*(a_k-a_k_inf)/(tau*tau_k) : 1
da_sd/dt = -phi*(a_sd- 1/(1+exp(-0.09*(V + 40))))/(tau*tau_sd) : 1
da_sr/dt = phi*(-alpha*p*g_sd*a_sd*(V - V_sd) - beta*a_sr)/(tau*tau_sr) : 1 
a_k_inf = 1/(1+exp(-0.25*(V + 25))) : 1
a_na = 1/(1+exp(-0.25*(V + 25))): 1
a_sd_inf = 1/(1+exp(-0.09*(V + 40))) : 1
phi : 1
p : 1
temp : 1
'''
#temp se agrega para simplificar el grafico.
G = NeuronGroup(6, eqs)#, threshold = 'V>-20e-3*volt',refractory = 'V >= -20*mV')
G.phi = phi_o
G.p = p_o
G.temp = Temp
statemon = StateMonitor(G,'V',record=True)
run(30000*ms)

fig = plt.figure(figsize=(8,8))
l = len(statemon.t)
for i in range(0,6):
    plt.subplot(6,1,i+1)
    plt.yticks([])
    if i !=5:
        plt.xticks([])        
    plt.plot(np.array(statemon.t[l//2:])-15, statemon.V[i][l//2:],linewidth = 0.3)
    ylabel('T = ' + str(G.temp[i]))  
xlabel('Time (s)')
plt.savefig("Temp.png" )