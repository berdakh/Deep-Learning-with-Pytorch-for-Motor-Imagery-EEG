# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import copy
# %%
import pickle
import time

# %%
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython import get_ipython
from pandas.plotting import table
from torch.utils.data import DataLoader, TensorDataset

from nu_smrutils import (CNN2D, get_data_loaders, load_pooled, loaddat, train_time)

# %% [markdown]
# ## Train time estimate 7
# %%
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if dev.type == 'cuda':
   print('Your GPU device name :', torch.cuda.get_device_name())        
else:
   print('No GPU Available :', dev)

# %%
dname = dict(BNCI2014004 = 'aBNCI2014004R.pickle',
             BNCI2014001 = 'aBNCI2014001R.pickle',
             Weibo2014   = 'aWeibo2014R.pickle',
             Physionet   = 'aPhysionetRR.pickle')

num_epochs = 150 
learning_rate = 1e-3
weight_decay = 1e-4  
batch_size = 64
verbose = 2

fs = 80
# define kernel size in terms of ms length 
time_window = 100 #ms
width = time_window*fs//1000    

# ker = 8 #timelength//chans 
h, w = 3, 1  #hight and width of a rectangular kernel      

# kernel size 
kernel_size = [(h, w*width), (h, w*width), (h, w*width), (h, w*width),
               (h, w*width), (h, w*width), (h, w*width), (h, w*width)]

conv_chan   = [1, 128, 64, 32, 16, 8]                     

# %%
# itenu_smrutilsis one of : ['BNCI2014004', 'BNCI2014001', 'Weibo2014', 'Physionet']
# itemname = 'BNCI2014004'
# filename = dname[itemname]
# iname    = itemname + '__'    
### Load pooled EEG data
#- the pooled data represents a set of combined subject-specific EEG data
alldat = dict()

for ii, itemname in enumerate(dname.keys()):
    data = loaddat(dname[itemname])    
    subjectIndex = list(range(108))
    class_name = ['left_hand', 'right_hand']
    
    datp = load_pooled(data, subjectIndex, class_name, normalize = True, test_size = 0.15)
    dat  = get_data_loaders(datp, batch_size = 64)
    
    input_size = (1, dat['test_data']['x_test'].shape[-2], 
                     dat['test_data']['x_test'].shape[-1])

    # Define the architecture
    model = CNN2D(input_size    = input_size, 
                  kernel_size   = kernel_size, 
                  conv_channels = conv_chan, 
                  dense_size    = 256, 
                  dropout       = 0.5)   
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                 weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    model.to(dev)  
    criterion.to(dev)       
   
    ttime = train_time(model, dat['dset_loaders'], dat['dset_sizes'],
                       criterion, optimizer, dev, num_epochs = num_epochs)
    
    alldat[itemname] = ttime 

    filename = itemname+'.pickle'

    with open(filename, 'wb') as ffile:
        pickle.dump(ttime, ffile)  


fname = 'train_time_alldat.pickle'
with open(fname, 'wb') as ffile:
    pickle.dump(alldat, ffile) 

# %% [markdown]
# ### Training time visualization 
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 5),
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize':'large',
          'ytick.labelsize':'large'}

pylab.rcParams.update(params)
plt.rcParams['figure.constrained_layout.use'] = False

"""
WEIBO2014
PhysioNet 
BCI Data-2a  
BCI Data-2b  
"""
dname = dict(BNCI2014004 = 'aBNCI2014004R.pickle',
             BNCI2014001 = 'aBNCI2014001R.pickle',
             Weibo2014   = 'aWeibo2014R.pickle',
             Physionet   = 'aPhysionetRR.pickle')            
# %%
d = loaddat('train_time_alldat.pickle')
d.keys()

# %%
def plotbar(dname, d):    
    f, axes = plt.subplots(1, 2)   
    datname = dname
    
    if dname   == "BNCI2014004":
        datname = 'BCI-DataSet2B'
    elif dname == "BNCI2014001":
        datname = 'BCI-DataSet2A'
             
    dat = pd.DataFrame(d[dname]['time_elapsed_epoch'])
    total_time = str(np.round(d[dname]['total_time_elapsed']))
    
    sns_plot = sns.boxplot(data=dat, ax = axes[0]) 
    if datname == 'BCI-DataSet2A':
        sns_plot.set_ylim(ymin=0, ymax=2)  
    sns_plot.set_xlabel('Time elapsed for each Epoch', fontsize = 15)
    sns_plot.set_ylabel('Time (seconds)')
    sns_plot.figure.tight_layout()
    
   # table(axes[0], np.round(dat.describe(), 2),
  #        loc='lower right', colWidths=[0.2, 0.2, 0.2])
    
    ttime = datname + " ::  Total training time elapsed for 150 epochs = " + str(total_time) + ' secs'
    #plt.text(0.001, 0.001, ttime, horizontalalignment='right', 
     #   size='medium', color='black')
    f.suptitle(ttime, x = 0.5, y = 1.04, fontsize = 15)
    
    dat = pd.DataFrame(d[dname]['time_elapsed_batch'])
    sns_plot = sns.boxplot(data=dat, ax = axes[1]) 
    #sns_plot.set_ylim(ymax=0.2)  
    sns_plot.set_xlabel('Time elapsed for each Batch', fontsize = 15)
    sns_plot.set_ylabel('Time (seconds)')
    sns_plot.figure.tight_layout()     
    #table(axes[1], np.round(dat.describe(), 2),
   #   loc='lower right', colWidths=[0.2, 0.2, 0.2])
    
    figname = dname +'.svg'
    plt.savefig(figname)

# %%
dname = 'BNCI2014004'
plotbar(dname, d)

# %%
dname = 'BNCI2014001'
plotbar(dname, d)

# %%
dname = 'Weibo2014'
plotbar(dname, d)

# %%
dname = 'Physionet'
plotbar(dname, d)

# %%
d.keys()

# %%
np
# %%
ttime

# %%
ttime = dict()

ttime['BCI-DataSet2A'] = [np.round(d['BNCI2014001']['total_time_elapsed'])]
ttime['BCI-DataSet2B'] = [np.round(d['BNCI2014004']['total_time_elapsed'])]
ttime['Weibo2014']     = [np.round(d['Weibo2014']['total_time_elapsed'])]
ttime['Physionet']     = [np.round(d['Physionet']['total_time_elapsed'])]

# %%
xtime = pd.DataFrame(ttime)

# %%
xtime

# %%
sns.set_context("paper")
style.use('seaborn-ticks')

params = {'legend.fontsize': 'large',
          'figure.figsize': (6.5, 3.5),
         'axes.labelsize': 'large',
         'axes.titlesize': 'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)
plt.rcParams['figure.constrained_layout.use'] = False

sns_plot = sns.barplot(data=xtime) 
sns_plot.set_xlabel('Datasets', fontsize = 14)
sns_plot.set_ylabel('Time (seconds)')
sns_plot.figure.tight_layout()
plt.savefig('barp.svg')

# %%
sns_plot = sns.barplot(data=xtime) 
sns_plot.set_xlabel('Datasets', fontsize = 15)
sns_plot.set_ylabel('Time Elapsed (seconds)')

def train_time(model, dset_loaders, dset_sizes, criterion, 
               optimizer, dev,  num_epochs = 150):  
    """This function give estimate of training time taken for a specific model
    
    """

    best_model, best_acc = model, 0.0 
    train_losses, val_losses, train_accs = [],[],[]
    val_accs, train_labels, val_labels   = [],[],[]

    # epoch time 
    train_start_time = time.time()
    time_elapsed_epoch = []

    for epoch in range(num_epochs):    
      epoch_start_time = time.time()  

      print('Epoch {}/{}'.format(epoch+1, num_epochs))          
      #### TRAIN PHASE ####
      ypred_labels, ytrue_labels = [], []   
      model.train(True)      
      running_loss, running_corrects = 0.0, 0.0  

      time_elapsed_batch = []     
      #### loop across batches ####
      for batch, datum in enumerate(dset_loaders['train']):          
          # first batch etc 
          inputs, labels = datum # get data 
          optimizer.zero_grad() # zero gradient         

          if not batch % 100:
            print('Processing batch: {} / Data : {}:'.format(batch, inputs.shape))

          batch_start_time = time.time() # record time                 
          preds= model(inputs.to(dev))   # make the prediction      

          loss = criterion(preds, labels.type(torch.LongTensor).to(dev)) # Calculate loss          
          loss.backward() # Backpropogate       
          optimizer.step()# Update the weights          

          time_elapsed_batch.append(time.time() - batch_start_time) # record time

          with torch.no_grad():
              # storch batch training performance 
              running_loss     += float(loss.item())                  
              running_corrects += torch.sum(preds.data.max(1)[1] == labels.data.to(dev))           
              ytrue_labels.append(labels.data.cpu().detach().numpy())
              ypred_labels.append(preds.data.max(1)[1].cpu().detach().numpy())        
              del loss  

      # ********* get the epoch loss and accuracy *********         
      epoch_loss = running_loss / dset_sizes['train']
      epoch_acc  = running_corrects.cpu().numpy()/dset_sizes['train']     

      train_losses.append(epoch_loss)
      train_accs.append(epoch_acc) 
      train_labels.append(dict(ypred = ypred_labels, ytrue = ytrue_labels))                  

      # record each epoch time. 
      time_elapsed_epoch.append(time.time() - epoch_start_time)  

    # record total train time     
    total_time_elapsed = time.time() - train_start_time  

    # save time info
    traintime = dict()
    traintime['filename'] = itemname 
    traintime['time_elapsed_epoch'] = time_elapsed_epoch
    traintime['time_elapsed_batch'] = time_elapsed_batch
    traintime['total_time_elapsed'] = total_time_elapsed
   
    return traintime     

model.train(False)     
for batch, vdatum in enumerate(dset_loaders['val']): 
    inputs, labels = vdatum
    if not batch % 10:
      print('Validate batch: {} / Data : {}:'.format(batch, inputs.shape))       

    preds = model(inputs.to(dev))  # predict and calculate the loss         
    loss  = criterion(preds, labels.type(torch.LongTensor).to(dev))               
    
    running_loss     += float(loss.item())                  
    running_corrects += torch.sum(preds.data.max(1)[1] == labels.data.to(dev))           
    ytrue_labels.append(labels.data.cpu().detach().numpy())
    ypred_labels.append(preds.data.max(1)[1].cpu().detach().numpy())              
    del loss     
  vepoch_loss = running_loss / dset_sizes['val'] 
  vepoch_acc  = running_corrects.cpu().numpy()/dset_sizes['val']  

  if  vepoch_acc > best_acc:
      best_acc, best_epoch = vepoch_acc, epoch 
      best_model = copy.deepcopy(model)           

  val_losses.append(vepoch_loss)
  val_accs.append(vepoch_acc) 
  val_labels.append(dict(ypred = ypred_labels, ytrue = ytrue_labels))                  

def get_best_epoch_labels(train_labels, best_epoch):
    """This function is used by train_model to store best epoch labels"""
    import numpy as np
    
    for jj in range(len(train_labels[best_epoch]['ypred'])-1):    
        if jj == 0: 
          ypred = train_labels[best_epoch]['ypred'][jj]              
          ytrue = train_labels[best_epoch]['ytrue'][jj] 
            
        ypred = np.concatenate([ypred, train_labels[best_epoch]['ypred'][jj+1]])
        ytrue = np.concatenate([ytrue, train_labels[best_epoch]['ytrue'][jj+1]])            
    return ypred, ytrue

time_elapsed = time.time() - start_time  

ytrain_best = get_best_epoch_labels(train_labels, best_epoch)
yval_best   = get_best_epoch_labels(val_labels,   best_epoch)      

info = dict(ytrain = ytrain_best, yval = yval_best, 
          best_epoch = best_epoch, best_acc = best_acc)

if verbose > 0:
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, 
                                                      time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc)) 
  print('Best Epoch :', best_epoch+1) 

#return best_model, train_losses, val_losses, train_accs, val_accs, info
