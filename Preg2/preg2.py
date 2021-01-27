def filtro(don,im,x,y):
    hlf = (len(don)-1)//2
    im[max(0,y-hlf):min(len(im),y+hlf+1),max(0,x-hlf):min(len(im[0]),x+hlf+1)] = don[max(0,hlf-y):min(len(don),hlf+len(im)-y),max(0,hlf-x):min(len(don),hlf+len(im[0])-x)]
    return im
def Don(x,y,vc,vs):
    return 1/(exp((x**2 + y**2)/(2*(vc**2))) * (2*pi*vc**2)) - 1/(exp((x**2 + y**2)/(2*vs**2))*(2*pi*(vs**2)))

def llaves_ordenadas(diccionario):
    diccionario = diccionario['t']
    l = [(j[0],i) for i,j in diccionario.items() if len(j)>0]
    l.sort()
    return [i for _,i in l]

    from brian2 import *
from scipy import signal, misc
from PIL import Image
import numpy as np
vc1 = 1.5
vs1 = 3.0
vc2 = 3.0
vs2 = 6.0
#name = 'g_tiger.jpg'
name = 'g_beach.jpg'
#name = 'g_desert.jpg'
#name = 'g_face.jpeg'
#name = 'g_cathedral.jpg'
#name = 'g_forest.jpg'
#name = 'g_savana.jpeg'
#name = 'g_traffic.jpg'
image = imread(name)

###########################
## Definicion de Filtros ##
###########################
size1 = 15#51#21
size2 = 33#93#43
don1 = np.zeros((size1,size1))
don2 = np.zeros((size2,size2))
for i in range (0,size1): # Se construye el kernel D para neuronas on-off del primer tamaño
    for j in range(0,size1):
        don1[i][j] = Don(-np.floor(size1/2) + j,-np.floor(size1/2) + i, vc1, vs1)
doff1 = -don1
for i in range (0,size2): # Se construye otro kernel D para neuronas con el segundo campo receptivo
    for j in range (0,size2):
        don2[j,i] = Don(-np.floor(size2/2) + j,-np.floor(size2/2) + i, vc2, vs2)
doff2 = -don2
I_1 = signal.convolve2d(image, don1, mode='same',boundary = 'symm') #Corriente campo receptivo on-off 1
I_2 = signal.convolve2d(image, don2, mode='same', boundary = 'symm')	#Corriente campo receptivo on-off 2


###########################
##Def. Grupos de neuronas##
###########################
tau = 1*second
eqs = '''
dv/dt = (I - v)/tau  : 1 
I : 1
x : 1
y : 1
g : 1
'''
G = NeuronGroup(6144,eqs,threshold='v > 5', reset='v= 0',refractory = 5*ms,
                 method ='exact' ,dt = 10**-5*second)
##Grupos de neuronas para campos receptivos On-off ##
G1on = G[0:2048]#NeuronGroup(4096, eqs, threshold='v > 20',
       #         reset='v = 0',refractory = 5*ms, method = 'exact',dt = 10**-6*second)
G2on = G[2048:3072]#NeuronGroup(2048, eqs, threshold='v > 20',
       #         reset='v = 0',refractory = 5*ms,method = 'exact', dt = 10**-6*second)
##Grupos de neuronas para campos receptivos Off-on
G1off = G[3072:5120]#NeuronGroup(4096, eqs, threshold='v > 20',
        #        reset='v = 0',refractory = 5*ms, method = 'exact',dt = 10**-6*second)
G2off = G[5120:6144]#NeuronGroup(2048, eqs, threshold='v > 20',
        #        reset='v = 0',refractory = 5*ms,method = 'exact', dt = 10**-6*second)
G.v = numpy.zeros((1,6144))

############################
##Asig. Parametros Neur.####
############################
#np.random.seed(1)

for i in range(0,2048):
    x = int(np.random.uniform(0,199))
    y = int(np.random.uniform(0,199))
    G1on.I[i] = I_1[y][x]
    G1on.x[i] = x
    G1on.y[i] = y
    G1on.g[i] = 1
for i in range(0,2048):
    x = int(np.random.uniform(0,199))
    y = int(np.random.uniform(0,199))
    G1off.I[i] = I_1[y][x]
    G1off.x[i] = x
    G1off.y[i] = y
    G1off.g[i] = 3
for i in range(0,1024):
    x = int(np.random.uniform(0,199))
    y = int(np.random.uniform(0,199))
    G2on.I[i] = I_2[y][x]
    G2on.x[i] = x
    G2on.y[i] = y
    G2on.g[i] = 2
for i in range(0,1024):
    x = int(np.random.uniform(0,199))
    y = int(np.random.uniform(0,199))
    G2off.I[i] = I_2[y][x]
    G2off.x[i] = x
    G2off.y[i] = y
    G2off.g[i] = 4
spk_trace = SpikeMonitor(G)

run(2000*ms)#Se usa este tiempo para asegurar que disparen las 600 neuronas requeridas
g_spikes = spk_trace.all_values()
order_spikes = llaves_ordenadas(g_spikes)

#####################
###Armado de tabla###
#####################

LUT = np.empty(len(order_spikes))       
N_s = len(order_spikes)
for i in range(0,len(order_spikes)):
    LUT[i] = (N_s - i)*255

im_aux = np.ones([200,200])*128
aux = np.zeros([200,200])
r = 600#len(order_spikes)
im_pos = 1
fig = plt.figure(figsize=(8,8))
for i in range(0,r):
    n = order_spikes[i]
    m = G.g[n]
    x = int(G.x[n])
    y = int(G.y[n])    
    aux[y][x] = 1
    if m == 1.0: #neuron on-off tamaño 1
        im_aux = im_aux + filtro(don1,aux,x,y)*LUT[i]#signal.convolve2d(aux, don1, mode='same')*LTU[i]
    elif m == 2.0: #neurona on-off tamaño2
        im_aux = im_aux + filtro(don2,aux,x,y)*LUT[i]#signal.convolve2d(aux, don2, mode='same')*LTU[i]
    elif m == 3.0: # neurona off-on tamaño1
        im_aux = im_aux - filtro(doff1,aux,x,y)*LUT[i]#signal.convolve2d(aux, don1, mode='same')*LTU[i]
    elif m == 4.0:
        im_aux = im_aux - filtro(doff2,aux,x,y)*LUT[i]#signal.convolve2d(aux, don2, mode='same')*LTU[i]
    j = i + 1
    if j == 5 or j == 10 or j == 100 or j == 300 or j == 600:       
        plt.subplot(1,6,im_pos)
        plt.yticks([])
        plt.xticks([])
        xlabel("N =" + str(j))
        imshow(im_aux,cmap = 'gray')
        im_pos +=1
    aux = aux*0
plt.subplot(1,6,6)
plt.yticks([])
plt.xticks([])
xlabel("Imagen original")
imshow(image, cmap = 'gray')
plt.savefig("a_" + name ) 

##Generacion Random 
im_aux = np.ones([200,200])*128
aux = np.zeros([200,200])
#im = np.zeros([200,200], dtype=np.uint8)
r = 600#len(order_spikes)
im_pos = 1
fig = plt.figure(figsize=(8,8))
for i in range(0,r):
    n = int(np.random.uniform(0,len(order_spikes)))#desde 0 hasta la cant. max que disparan.
    m = G.g[n]
    x = int(G.x[n])
    y = int(G.y[n])    
    aux[y][x] = 1
    if m == 1.0: #neuron on-off tamaño 1
        im_aux = im_aux + filtro(don1,aux,x,y)*LUT[i]#signal.convolve2d(aux, don1, mode='same')*LTU[i]
    elif m == 2.0: #neurona on-off tamaño2
        im_aux = im_aux + filtro(don2,aux,x,y)*LUT[i]#signal.convolve2d(aux, don2, mode='same')*LTU[i]
    elif m == 3.0: # neurona off-on tamaño1
        im_aux = im_aux - filtro(doff1,aux,x,y)*LUT[i]#signal.convolve2d(aux, don1, mode='same')*LTU[i]
    elif m == 4.0:
        im_aux = im_aux - filtro(doff2,aux,x,y)*LUT[i]#signal.convolve2d(aux, don2, mode='same')*LTU[i]
    j = i + 1
    if j == 5 or j == 10 or j == 100 or j == 300 or j == 600:       
        plt.subplot(1,6,im_pos)
        plt.yticks([])
        plt.xticks([])
        xlabel("N =" + str(j))
        imshow(im_aux,cmap = 'gray')
        im_pos +=1
    aux = aux*0
plt.subplot(1,6,6)
plt.yticks([])
plt.xticks([])
xlabel("Imagen original")
imshow(image, cmap = 'gray') 
plt.savefig("b_" + name ) 
