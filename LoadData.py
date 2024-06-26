# -*- coding: utf-8 -*-
"""
LTL
Import datay from H5 files for analysis...

"""
def LoadData(filename):
    import h5py
    import numpy as np 
    
    phoibos = 0
    
    if phoibos == 0:
    #####################
        f = h5py.File(filename,'r')
        
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())
        
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]
        a_group_key2 = list(f.keys())[1]
        #print(a_group_key)
        # get the object type for a_group_key: usually group or dataset
        #print(type(f[a_group_key])) 
        
        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        
        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        data = list(f[a_group_key])
        data2 = list(f[a_group_key2])
        print('First Keys: ' + str(data))
        print('Second Keys: ' + str(data2))
        
        # preferred methods to get dataset values:
        #ds_obj = f[a_group_key] #returns as a h5py dataset object
        
        ax_E = f['axes/ax2'][()]  # returns as a numpy array
        #ax_E = f['axes/ax_E'][()]  # returns as a numpy array
        ax_E = ax_E.astype(np.float32)
        
        ax_kx = f['axes/ax0'][()]  # returns as a numpy array
        #ax_kx = f['axes/ax_kx'][()] 
        ax_kx = ax_kx.astype(np.float32)
        
        ax_ky = f['axes/ax1'][()]  # returns as a numpy array
        #ax_ky = f['axes/ax_ky'][()] 
        ax_ky = ax_ky.astype(np.float32)
        
        i_data = f['binned/BinnedData'][()]
    
        if  'delay' in list(f[a_group_key]):
            ax_ADC = f['axes/delay'][()]  # returns as a numpy array
            ax_ADC = ax_ADC.astype(np.float32)
            
        elif 'theta' in list(f[a_group_key]):
            ax_ADC = f['axes/theta'][()]  # returns as a numpy array
            ax_ADC = ax_ADC.astype(np.float32)
        
        elif 'ax3' in list(f[a_group_key]):
            ax_ADC = f['axes/ax3'][()]  # returns as a numpy array
            ax_ADC = ax_ADC.astype(np.float32)    
            
            I = np.zeros((len(ax_kx), len(ax_ky), len(ax_E), len(ax_ADC)), dtype='float32') # Initialize the data cube
            I = np.squeeze(I.astype(np.float32))
            
            #print(I.shape)
            #print(i_data.shape)
            I[:,:,:,:] = i_data #np.transpose(i_data, (3, 0, 1, 2)) 
    
        else:
            ax_ADC = np.zeros(1)
            I = np.zeros((len(ax_kx), len(ax_ky), len(ax_E)), dtype='float32') # Initialize the data cube
            
            I[:,:,:] = i_data #np.transpose(i_data, (3, 0, 1, 2))     
        
        print('')
        print('The data shape is...' + str(i_data.shape))
        print(filename + ' has been loaded... Happy Analysis...')
        
        return I, ax_kx, ax_ky, ax_E, ax_ADC
    elif phoibos == 1:
        
        f = h5py.File(filename,'r')

        print("Keys: %s" % f.keys())
        
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]
        a_group_key2 = list(f.keys())[1]
        #print(a_group_key)
        # get the object type for a_group_key: usually group or dataset
        #print(type(f[a_group_key])) 
        
        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        
        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        data = list(f[a_group_key])
        data2 = list(f[a_group_key2])
        print('First Keys: ' + str(data))
        print('Second Keys: ' + str(data2))