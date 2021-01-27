from brian2 import *
import numpy as np
A = 10
v = 1
def gauss(x,c):
    return 10*exp(-(x-c)**2/2)
%matplotlib inline
ui = np.linspace(-10,10,20)#np.random.uniform(-10.0,10.0,20)

t= np.arange(-20,20,0.1) 
fig = plt.figure(figsize=(8,8))
for i in range (0,20):
    plot(t,gauss(t,ui[i]))#plot(t,10*exp(-(t-ui[i])**2/2))
plt.savefig("sel_a.jpg")

x = np.random.uniform(-5.0,5.0,10)
r = np.zeros((10,20))
for i in range(0, 10):
    for j in range(0,20):
        r[i][j] = gauss(x[i],ui[j])
        #print(gauss(x,ui[j]))
#r[x][y] respuesta para el estimulo x de la neurona y 

#Winner Take-All Decoding
j = np.zeros((len(r)))
s_wta = np.zeros(len(x))
for i in range(len(r)):
    j[i] = np.where(r[i]==np.max(r[i]))[0]
    s_wta[i] = ui[int(j[i])]

 #Center of mass decoding
s_com = np.zeros(len(x))
r_sum = np.sum(r, axis = 1)#Suma de respuestas de cada neurona a el estimulo
sr_sum= np.sum(ui*r, axis = 1)
s_com = sr_sum/r_sum

#Maximum likelihood decoding
s_ml = np.zeros(len(x))
for i in range(len(x)):
    aux = r[i]*np.log(gauss(x[i],ui))
    for j in range(len(aux)):
        pos = np.where(aux==np.max(aux))[0]
    s_ml[i] = ui[pos]

print("Corr WTA:" + str(np.corrcoef(x,s_wta)[0][1]))
print("Corr COM:" + str(np.corrcoef(x,s_com)[0][1]))
print("Corr ML:" + str(np.corrcoef(x,s_ml)[0][1]))