#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:49:10 2024

@author: lawsonlloyd

"""




#%%

#7924, 7933

edc_t = np.loadtxt("Data_trEDCs_7933.txt", skiprows = 1)

temp_axis = np.loadtxt("Data_t_7933.txt", skiprows = 1)

energy_axis = 17.58 + np.arange(0,344)*0.00767442
#%%

%matplotlib inline

plt.figure(figsize=(10,7))

Ts = [120, 125, 130, 135, 145, 155, 165, 175, 185, 200, 225, 250, 300]
Ts = [120, 140, 160, 180, 200, 240, 300]
Ts = temp_axis
#Ts = Ts[::-1]

num_lines_total = len(temp_axis)
num_lines = len(Ts)

colors = plt.cm.coolwarm(np.linspace(0,1,num_lines_total))
#colors = colors[::-1]

colors = plt.cm.coolwarm(np.linspace(0, 1, 181)) 
cbar_temp = np.linspace(120, 300, 181)

cm = plt.cm.ScalarMappable(norm = plt.Normalize(vmin=temp_axis[0],vmax=temp_axis[-1]), cmap=plt.cm.coolwarm)

#for i in range(num_lines-1, -1, -1):
for i in range(0, num_lines, 1):

    j = (np.abs(temp_axis - Ts[i])).argmin()
    j_temp = (np.abs(Ts[i]-cbar_temp)).argmin()

    plt.plot(energy_axis, edc_t[:,j], label = str(temp_axis[j]), color=colors[j_temp])
    #print(i)
    
cbar = plt.colorbar(cm)
cbar.set_label('T, K', rotation=90, fontsize=22)
#plt.legend()                                                
plt.xlim(17.8,20.2)
plt.xlabel('Energy, eV', fontsize=22)
plt.ylabel('Int., arb. units.', fontsize=22)
plt.tick_params(axis='both', labelsize=20)
cbar.ax.tick_params(labelsize=20)

#fig.tight_layout()
plt.show()

#%%

%matplotlib inline

plt.figure(figsize=(10,7))

Ts = [120, 125, 130, 135, 145, 155, 165, 175, 185, 200, 225, 250, 300]
Ts = [120, 140, 160, 180, 200, 220, 300]
#Ts = temp_axis
#Ts = Ts[::-1]

num_lines_total = len(temp_axis)
num_lines = len(Ts)

colors = plt.cm.coolwarm(np.linspace(0,1, 181)) 
cbar_temp = np.linspace(120, 300, 181)

#colors = colors[::-1]
cm = plt.cm.ScalarMappable(norm = plt.Normalize(vmin=temp_axis[0],vmax=temp_axis[-1]), cmap=plt.cm.coolwarm)

for i in range(num_lines-1, -1, -1):
    j = (np.abs(temp_axis - Ts[i])).argmin()
    j_temp = (np.abs(Ts[i]-cbar_temp)).argmin()

    plt.plot(energy_axis, (edc_t[:,j]/3e8)+0.15*i, label = str(temp_axis[j]), color=colors[j_temp])

    #plt.plot(energy_axis, (edc_t[:,j]/max(edc_t[:,j]))+0.15*i, label = str(temp_axis[j]), color=colors[j])
    #print(i)
    
cbar = plt.colorbar(cm)
cbar.set_label('T, K', rotation=90, fontsize=22)
cbar.ax.tick_params(labelsize=20)

#plt.legend()                                                
plt.xlim(17.8,20.2)
plt.ylim(0, 1.8)
plt.xlabel('Energy, eV', fontsize=22)
plt.ylabel('Int., arb. units.', fontsize=22)
plt.tick_params(axis='both', labelsize=20)

#fig.tight_layout()
plt.show()


#%%

%matplotlib inline

plt.figure()
plt.plot(energy_axis, edc_t[:,18], label = str(temp_axis[18]))
plt.plot(energy_axis, edc_t[:,25], label = str(temp_axis[25]))

plt.plot(energy_axis, edc_t[:,29], label = str(temp_axis[29]))
plt.plot(energy_axis, edc_t[:,-1], label = str(temp_axis[-1]))

plt.legend()                                                
plt.xlim(17.8,20.2)
plt.xlabel('KE, eV')
plt.ylabel('Int., arb. units.')

#%%


