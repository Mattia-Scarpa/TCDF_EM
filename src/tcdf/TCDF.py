from datetime import date
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from .model import ADDSTCN
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import numpy as np
import heapq
import copy
import os
import sys

def preparedata(file, target):
    """Reads data from csv file and transforms it to two PyTorch tensors: dataset x and target time series y that has to be predicted."""
    df_data = pd.read_csv(file)
    
    df_y = df_data.copy(deep=True)[[target]]
    df_x = df_data.copy(deep=True)
    
    df_yshift = df_y.copy(deep=True).shift(periods=1, axis=0)
    df_yshift[target]=df_yshift[target].fillna(0.)
    df_x[target] = df_yshift *0 # remove self causation
    data_x = df_x.values.astype("float32").transpose()    
    data_y = df_y.values.astype("float32").transpose()
    data_x = torch.from_numpy(data_x).to(device="cuda")
    data_y = torch.from_numpy(data_y).to(device="cuda")

    x, y = Variable(data_x), Variable(data_y)
    return x, y


def train(epoch, traindata, traintarget, modelname:ADDSTCN, optimizer,log_interval,epochs):
    """Trains model by performing one epoch and returns attention scores and loss."""

    modelname.train()
    x, y = traindata[0:1], traintarget[0:1]
        
    optimizer.zero_grad()
    epochpercentage = (epoch/float(epochs))*100
    output = modelname(x)

    attentionscores = modelname.fs_attention_logits
    
    loss_mes = F.mse_loss(output, y)
    loss_lasso = modelname.attention_regularization()
    loss = loss_mes + loss_lasso
    
    loss.backward()
    optimizer.step()
    
    modelname.lasso_lambda = modelname.lasso_lambda * 0.9999 #decay lasso lambda to zero

    if epoch % log_interval ==0 or epoch % epochs == 0 or epoch==1:
        print("Epoch: {:2d} [{:.0f}%] \tLoss: {:.6f}".format(epoch, epochpercentage, loss))

    return attentionscores.data, loss

def findcauses(target, cuda, epochs, kernel_size, layers, 
               log_interval, lr, optimizername, seed, dilation_c, significance, file):
    """Discovers potential causes of one target time series, validates these potential causes with PIVM and discovers the corresponding time delays"""

    print("\n", "Analysis started for target: ", target)
    torch.manual_seed(seed)
    
    X_train, Y_train = preparedata(file, target)
    X_train = X_train.unsqueeze(0).contiguous()
    Y_train = Y_train.unsqueeze(2).contiguous()

    input_channels = X_train.size()[1]
       
    targetidx = pd.read_csv(file).columns.get_loc(target)
          
    model = ADDSTCN(targetidx, input_channels, layers, kernel_size=kernel_size, cuda=cuda, dilation_c=dilation_c)
    if cuda:
        model.cuda()
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()

    optimizer = getattr(optim, optimizername)(model.parameters(), lr=lr)    
    
    scores, firstloss = train(1, X_train, Y_train, model, optimizer,log_interval,epochs)
    firstloss = firstloss.cpu().data.item()
    for ep in range(2, epochs+1):
        scores, realloss = train(ep, X_train, Y_train, model, optimizer,log_interval,epochs)
    realloss = realloss.cpu().data.item()
    
    print(scores)
    
    potentials = [idx for idx, s in enumerate(scores) if s > .3]
    print("Potential causes: ", potentials)
    validated = copy.deepcopy(potentials)
    
    #Apply PIVM (permutes the values) to check if potential cause is true cause
    for idx in potentials:
        random.seed(seed)
        X_test2 = X_train.clone().cpu().numpy()
        random.shuffle(X_test2[:,idx,:][0])
        shuffled = torch.from_numpy(X_test2)
        if cuda:
            shuffled=shuffled.cuda()
        model.eval()
        output = model(shuffled)
        testloss = F.mse_loss(output, Y_train)
        testloss = testloss.cpu().data.item()
        
        diff = firstloss-realloss
        testdiff = firstloss-testloss

        if testdiff>(diff*significance): 
            validated.remove(idx)
    
 
    weights = []
    
    #Discover time delay between cause and effect by interpreting kernel weights
    for layer in range(layers):
        weight = model.dwn.network[layer].net[0].weight.abs().view(model.dwn.network[layer].net[0].weight.size()[0], model.dwn.network[layer].net[0].weight.size()[2])
        weights.append(weight)

    causeswithdelay = dict()    
    for v in validated: 
        totaldelay=0    
        for k in range(len(weights)):
            w=weights[k]
            row = w[v]
            twolargest = heapq.nlargest(2, row)
            m = twolargest[0]
            m2 = twolargest[1]
            if m > m2:
                index_max = len(row) - 1 - max(range(len(row)), key=row.__getitem__)
            else:
                #take first filter
                index_max=0
            delay = index_max *(dilation_c**k)
            totaldelay+=delay
        if targetidx != v:
            causeswithdelay[(targetidx, v)]=totaldelay
        else:
            causeswithdelay[(targetidx, v)]=totaldelay+1
    print("Validated causes: ", validated)
    
    return validated, causeswithdelay, realloss, scores.view(-1).cpu().detach().numpy().tolist()


# ============== NEW FUNCTIONS TO HANDLE MULTI CSV LIST ==============


class MultiCSVDataset(Dataset):
    """Dataset class for handling multiple CSV files"""
    def __init__(self, csv_files, target_column):
        self.csv_files = csv_files
        self.target_column = target_column
        self.data_list = []
        self.file_indices = []
        
        # Load all CSV files and prepare data
        for file_idx, csv_file in enumerate(csv_files):
            df = pd.read_csv(csv_file)
            if target_column in df.columns:
                x, y = preparedata(csv_file, target_column)
                self.data_list.append((x, y, csv_file))
                self.file_indices.append(file_idx)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]


def train_multi(epoch, dataloader, modelname, optimizer, log_interval, epochs):
    modelname.train()
    total_loss = 0
    all_attention_scores = []
    
    # Verifica che il modello sia su GPU
    model_device = next(modelname.parameters()).device
    if epoch == 1:
        print(f"Model device: {model_device}")
    
    for batch_idx, (x, y, filename) in enumerate(dataloader):
        x = x.contiguous()
        y = y.permute([0,2,1]).contiguous()
        
        # Assicurati che i dati siano su GPU
        if modelname.cuda and torch.cuda.is_available():
            x = x.to(model_device, non_blocking=True)
            y = y.to(model_device, non_blocking=True)
        
        optimizer.zero_grad()
        output = modelname(x)
        
        attentionscores = modelname.fs_attention_logits
        all_attention_scores.append(attentionscores.detach())
        
        loss_mes = F.mse_loss(output, y)
        loss_lasso = modelname.attention_regularization()
        loss = loss_mes + loss_lasso
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    modelname.lasso_lambda = modelname.lasso_lambda * 0.9999  # decay lasso lambda
    
    avg_loss = total_loss / len(dataloader)
    epochpercentage = (epoch / int(epochs)) * 100
    
    print(f"Epoch: {epoch}/{float(epochs)} [{epochpercentage:.2f}%]", end="\r")
    
    if epoch % log_interval == 0 or epoch % epochs == 0 or epoch == 1:
        print("Epoch: {:2d} [{:.0f}%] \tAvg Loss: {:.6f}".format(epoch, epochpercentage, avg_loss))
    
    # Return average attention scores
    if all_attention_scores:
        avg_scores = torch.mean(torch.stack(all_attention_scores), dim=0)
        return avg_scores, avg_loss
    
    return None, avg_loss


def findcauses_multi(target, cuda, epochs, kernel_size, layers, 
                     log_interval, lr, optimizername, seed, dilation_c, significance, csv_files):
    """Discovers potential causes across multiple CSV files"""
    
    print("\n", "Analysis started for target: ", target)
    torch.manual_seed(seed)
    
    
    # Create dataset and dataloader
    dataset = MultiCSVDataset(csv_files, target)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    if len(dataset) == 0:
        print(f"Target {target} not found in any CSV files")
        return [], {}, 0, []
    
    # Get input channels and target index from first sample
    first_x, _, first_file = dataset[0]
    input_channels = first_x.size()[-2]
    
    df_first = pd.read_csv(first_file)
    targetidx = df_first.columns.get_loc(target)
    
    model = ADDSTCN(targetidx, input_channels, layers, kernel_size=kernel_size, 
                    cuda=cuda, dilation_c=dilation_c)
    if cuda:
        model.cuda()
    
    optimizer = getattr(optim, optimizername)(model.parameters(), lr=lr)
    
    # Training loop
    scores = None
    firstloss = None
    realloss = None
    
    for ep in range(1, epochs + 1):
        scores, loss = train_multi(ep, dataloader, model, optimizer, log_interval, epochs)
        if ep == 1:
            firstloss = loss
        realloss = loss
    
    if scores is None:
        return [], {}, 0, []
    
    print("Attention scores:\n[", end="")
    for i, s in enumerate(scores):
        print(f"{s.item():.2e}{str(", " if i < (len(scores)-1) else "")}", end="")
    print("]")    
    potentials = [idx for idx, s in enumerate(scores) if s.abs() > 0.2]
    print("Potential causes: ", potentials)
    # Validate causes
    validated = copy.deepcopy(potentials)
    
    # Apply PIVM validation (simplified for multiple files)
    improvement_ratio = (float(firstloss)/float(realloss))
    print("="*25)
    print(f"- first_loss: {firstloss:.4e}\n - final_loss: {realloss:.4e}\nImprovement ratio: {improvement_ratio:.4f}")
    print("."*25)
    for idx in potentials:
        # For simplicity, validate on first file
        X_test, Y_test = preparedata(dataset.data_list[0][2], target)
        X_test = X_test.unsqueeze(0).contiguous()
        Y_test = Y_test.unsqueeze(2).contiguous()
        
        if cuda:
            X_test = X_test.cuda()
            Y_test = Y_test.cuda()
        
        random.seed(seed)
        X_test2 = X_test.clone().cpu().numpy()
        X_test2[:,targetidx,:] = X_test2[:,targetidx,:]*0.
        random.shuffle(X_test2[:,idx,:][0])
        shuffled = torch.from_numpy(X_test2)
        if cuda:
            shuffled = shuffled.cuda()
        
        model.eval()
        output = model(shuffled)
        testloss = F.mse_loss(output, Y_test)
        testloss = testloss.cpu().data.item()
        
        # diff = firstloss - realloss
        # testdiff = firstloss - testloss
        improvement_ratio_test = (float(firstloss)/float(testloss))
        degradation = improvement_ratio/improvement_ratio_test
        print(f"Cause {df_first.columns[idx]}")
        print(f"- test_loss : {testloss:.4e}\nImprovement ratio: {improvement_ratio_test:.4f} - Degradation: {degradation}")
        if degradation > 10:
            print(f"OK! {df_first.columns[idx]} is a VALIDATED cause!")
        else:
            validated.remove(idx)
        print("-"*25)
        
    print("="*25)
    
    # Discover delays
    weights = []
    for layer in range(layers):
        weight = model.dwn.network[layer].net[0].weight.abs().view(
            model.dwn.network[layer].net[0].weight.size()[0], 
            model.dwn.network[layer].net[0].weight.size()[2])
        weights.append(weight)
    
    causeswithdelay = dict()
    for v in validated:
        totaldelay = 0
        for k in range(len(weights)):
            w = weights[k]
            row = w[v]
            twolargest = heapq.nlargest(2, row)
            m = twolargest[0]
            m2 = twolargest[1]
            if m > m2:
                index_max = len(row) - 1 - max(range(len(row)), key=row.__getitem__)
            else:
                index_max = 0
            delay = index_max * (dilation_c ** k)
            totaldelay += delay
        
        if targetidx != v:
            causeswithdelay[(targetidx, v)] = totaldelay
        else:
            causeswithdelay[(targetidx, v)] = totaldelay + 1
    
    print("Validated causes: ", validated)
    
    return validated, causeswithdelay, realloss, scores.view(-1).cpu().detach().numpy().tolist()
