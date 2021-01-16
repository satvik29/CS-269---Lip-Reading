from model import * 
from utils import * 
from dataset import *

import os
import argparse
from tqdm.notebook import tqdm # remove .notebook if not running script on jupyter notebook
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

import wandb 


def train_LipNet(device, model, train_gen, val_gen, criterion, optimizer, scheduler, options): 
    
    train_loss_history, train_acc_history = [], []
    val_loss_history, val_acc_history = [], []

    epochs = options["epochs"]

    for epoch in tqdm(range(epochs)):      
        print(f'Epoch {epoch}/{epochs}')
        print('-' * 30)
           
        running_train_loss, running_val_loss = 0, 0
        running_train_acc, running_val_acc = 0, 0

        # Training 
        iterations = 0
        for batch_idx,(inputs,labels) in enumerate(train_gen): 
            
            batch_size = inputs.shape[0]
            # print(f'shapes: input = {inputs.shape}, label = {labels.shape}')

            # Put model  in 'Training' Mode
            model.train()
            # zero the parameter gradients
            optimizer.zero_grad()

            # Transfer to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass           
            logits = model(inputs)

            # calculate loss
            loss = criterion(logits, labels)
            batch_train_loss = loss.item()
            running_train_loss += batch_train_loss

            # Back Propagation
            loss.backward()

            # Gradient Clipping
            #total_norm = 0
            #for p in model.parameters():
            #    param_norm = p.grad.data.norm(2)
            #    total_norm += param_norm.item() ** 2
            #total_norm = total_norm ** (1. / 2)
            #print(f'Gradients norm: {total_norm}')
            # _ = torch.nn.utils.clip_grad_norm_(network.parameters(), GRADIENTS_NORM)

            # Update Network Paramaters
            optimizer.step() 
           
            # Accuracy 
            _, predicted = torch.max(logits, 1)
            # print(predicted.detach().cpu().numpy())
            # print(labels.cpu().numpy())
            batch_train_acc = 100 * ((predicted == labels).sum().item()) / labels.size(0)
            running_train_acc += batch_train_acc

            # Stats and checkpoint
            if iterations % 100 == 0:
                print(f'Epoch: {epoch}/{epochs}, Iteration: {iterations}')
                print(f'Train Loss: {batch_train_loss}, Train Accuracy: {batch_train_acc}')
            
                #wandb.log({'iters': iterations + epoch*len(train_gen), 
                #           'running_train_loss': running_train_loss / (iterations+1),
                #          'running_train_accuracy': running_train_acc / (iterations+1)})

            #    currentTime = (datetime.strftime(datetime.now(),"%H_%M_%S_%d_%m_%Y"))
            #    checkpoint_path = os.path.join(options['checkpoint_dir'], f'{options["model"]}_epoch_{epoch}_iter_{iterations}_{currentTime}.pt')
                
            #    save_dict = {'epoch': epoch,
            #    'model_state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict()}
            #    for key in options.keys(): 
            #        save_dict[key] = options[key]

            #    torch.save(save_dict, checkpoint_path)
            #    print(f'Checkpoint saved at {checkpoint_path}')

            iterations += 1

            # Clear memory cache 
            torch.cuda.empty_cache()
        
        train_loss = running_train_loss / iterations
        train_acc = running_train_acc / iterations

        # Validation 
        iterations = 0
        for batch_idx, (inputs, labels) in enumerate(val_gen): 
            
            batch_size = inputs.shape[0]

            # Put model  in 'Eval' Mode
            model.eval()
 
            # Transfer to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(inputs)

            # calculate loss
            loss = criterion(logits, labels)
            running_val_loss += loss.item()
            
            # Accuracy 
            _, predicted = torch.max(logits, 1)
            batch_val_acc = 100 * ((predicted == labels).sum().item()) / labels.size(0)
            running_val_acc += batch_val_acc

            iterations += 1

            # Clear memory cache 
            torch.cuda.empty_cache()

        val_loss = running_val_loss / iterations
        val_acc = running_val_acc / iterations
        
        # Adjust the learning rate
        if scheduler is not []: 
            scheduler.step(val_loss)

        # history tracking
        print(f'End of epoch: {epoch}/{epochs}')
        print(f'Train Loss: {train_loss}, Val Loss: {val_loss}')
        print(f'Train Accuracy: {train_acc}, Val Accuracy: {val_acc}')
        
        wandb.log({'epoch': epoch, 
                   'train_loss': train_loss, 'val_loss': val_loss,
                   'train_acc': train_acc, 'val_acc': val_acc})
              
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # checkpoint every 5 epochs
        if epoch % 5 == 0: 
            currentTime = (datetime.strftime(datetime.now(),"%H_%M_%S_%d_%m_%Y"))
            checkpoint_path = os.path.join(options['checkpoint_dir'], f'{options["model"]}_EndOfEpoch_{epoch}_{currentTime}.pt')
            
            save_dict = {'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
            for key in options.keys(): 
                save_dict[key] = options[key]

            torch.save(save_dict, checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')
    
    history = {
        'train_loss': train_loss_history, 
        'train_acc': train_acc_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history,
        }

    return history

def evaluate_LipNet(device, model, test_gen):
   
    all_predictions, all_labels = [], [] 
    iterations, running_test_acc = 0, 0
    for batch_idx, (inputs, labels) in enumerate(test_gen): 
            
        batch_size = inputs.shape[0]

        # Put model  in 'Eval' Mode
        model.eval()
 
        # Transfer to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(inputs)
            
        # Accuracy 
        _, predicted = torch.max(logits, 1)
        batch_test_acc = 100 * ((predicted == labels).sum().item()) / labels.size(0)
        running_test_acc += batch_test_acc

        # store predictions and true labels
        all_predictions.append(predicted.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        iterations += 1

        # Clear memory cache 
        torch.cuda.empty_cache()
    
    test_acc = running_test_acc / iterations
    all_predictions = np.concatenate(all_predictions,axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return test_acc, all_predictions, all_labels

def train_features_model(device, model, train_gen, val_gen, criterion, optimizer, scheduler, options): 
    
    train_loss_history, train_acc_history = [], []
    val_loss_history, val_acc_history = [], []

    epochs = options["epochs"]

    for epoch in tqdm(range(epochs)):      
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 30)
           
        running_train_loss, running_val_loss = 0, 0
        running_train_acc, running_val_acc = 0, 0

        # Training 
        iterations = 0
        for batch_idx,(inputs,labels) in enumerate(train_gen): 
            
            # print(f'shapes: input = {inputs.shape}, label = {labels.shape}')

            # Put model  in 'Training' Mode
            model.train()
            # zero the parameter gradients
            optimizer.zero_grad()

            # Init hidden states for LSTM
            batch_size = inputs.shape[0]
            state_h, state_c = model.init_hidden(batch_size)

            # Transfer to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            state_h = state_h.to(device)
            state_c = state_c.to(device)

            # Forward pass           
            logits, (state_h, state_c) = model(inputs, (state_h, state_c))

            # calculate loss
            loss = criterion(logits, labels)
            batch_train_loss = loss.item()
            running_train_loss += batch_train_loss

            # Removes states from computational grad to preserve functionality of autograd
            state_h = state_h.detach()
            state_c = state_c.detach()

            # Back Propagation
            loss.backward()

            # Gradient Clipping
            # _ = torch.nn.utils.clip_grad_norm_(network.parameters(), GRADIENTS_NORM)

            # Update Network Paramaters
            optimizer.step() 

            # Accuracy 
            _, predicted = torch.max(logits, 1)
            # print(predicted.detach().cpu().numpy())
            # print(labels.cpu().numpy())
            batch_train_acc = 100 * ((predicted == labels).sum().item()) / labels.size(0)
            running_train_acc += batch_train_acc
            
            # Stats and checkpoint
            if iterations % 100 == 0:
                print(f'Epoch: {epoch}/{epochs}, Iteration: {iterations}')
                print(f'Train Loss: {batch_train_loss}, Train Accuracy: {batch_train_acc}')
            
                #wandb.log({'iters': iterations + epoch*len(train_gen), 
                #           'running_train_loss': running_train_loss / (iterations+1),
                #          'running_train_accuracy': running_train_acc / (iterations+1)})

                #currentTime = (datetime.strftime(datetime.now(),"%H_%M_%S_%d_%m_%Y"))
                #checkpoint_path = os.path.join(options['checkpoint_dir'], f'{options["model"]}_epoch_{epoch}_iter_{iterations}_{currentTime}.pt')
                
                #save_dict = {'epoch': epoch,
                #'model_state_dict': model.state_dict(),
                #'optimizer_state_dict': optimizer.state_dict()}
                #for key in options.keys(): 
                #    save_dict[key] = options[key]

                #torch.save(save_dict, checkpoint_path)
                #print(f'Checkpoint saved at {checkpoint_path}')                

            iterations += 1

            # Clear memory cache 
            torch.cuda.empty_cache()

        train_loss = running_train_loss / iterations
        train_acc = running_train_acc / iterations

        # Validation 
        iterations = 0
        for batch_idx, (inputs, labels) in enumerate(val_gen): 
            
            # Put model  in 'Eval' Mode
            model.eval()
            
            # Init hidden states for LSTM
            batch_size = inputs.shape[0]
            state_h, state_c = model.init_hidden(batch_size)

            # Transfer to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            state_h = state_h.to(device)
            state_c = state_c.to(device)

            # Forward pass
            logits, (state_h, state_c) = model(inputs, (state_h, state_c))

            # calculate loss
            loss = criterion(logits, labels)
            running_val_loss += loss.item()

            # Removes states from computational grad to preserve functionality of autograd
            state_h = state_h.detach()
            state_c = state_c.detach()    
            
            # Accuracy 
            _, predicted = torch.max(logits, 1)
            batch_val_acc = 100 * ((predicted == labels).sum().item()) / labels.size(0)
            running_val_acc += batch_val_acc

            iterations += 1

            # Clear memory cache 
            torch.cuda.empty_cache()

        val_loss = running_val_loss / iterations
        val_acc = running_val_acc / iterations
        
        # Adjust the learning rate
        if scheduler is not []: 
            scheduler.step(val_loss)

        # history tracking
        print(f'End of epoch: {epoch}/{epochs}')
        print(f'Train Loss: {train_loss}, Val Loss: {val_loss}')
        print(f'Train Accuracy: {train_acc}, Val Accuracy: {val_acc}')
        
        wandb.log({'epoch': epoch, 
                   'train_loss': train_loss, 'val_loss': val_loss,
                   'train_acc': train_acc, 'val_acc': val_acc})
              
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # checkpoint every 5 epochs
        if epoch % 5 == 0: 
            currentTime = (datetime.strftime(datetime.now(),"%H_%M_%S_%d_%m_%Y"))
            checkpoint_path = os.path.join(options['checkpoint_dir'], f'{options["model"]}_EndOfEpoch_{epoch}_{currentTime}.pt')
            
            save_dict = {'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
            for key in options.keys(): 
                save_dict[key] = options[key]

            torch.save(save_dict, checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')
    
    history = {
        'train_loss': train_loss_history, 
        'train_acc': train_acc_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history,
        }

    return history

def evaluate_features_model(device, model, test_gen): 

    all_predictions, all_labels = [], []

    iterations, running_test_acc = 0, 0
    for batch_idx, (inputs, labels) in enumerate(test_gen):          
        # Put model  in 'Eval' Mode
        model.eval()
            
        # Init hidden states for LSTM
        batch_size = inputs.shape[0]
        state_h, state_c = model.init_hidden(batch_size)

        # Transfer to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        # Forward pass
        logits, (state_h, state_c) = model(inputs, (state_h, state_c))

        # Removes states from computational grad to preserve functionality of autograd
        state_h = state_h.detach()
        state_c = state_c.detach()    
            
        # Accuracy 
        _, predicted = torch.max(logits, 1)
        batch_test_acc = 100 * ((predicted == labels).sum().item()) / labels.size(0)
        running_test_acc += batch_test_acc

        # store predictions and true labels
        all_predictions.append(predicted.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        iterations += 1

        # Clear memory cache 
        torch.cuda.empty_cache()
    
    test_acc = running_test_acc / iterations
    all_predictions = np.concatenate(all_predictions,axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return test_acc, all_predictions, all_labels

if __name__ == "__main__": 

    # Arguement parser #
    parser = argparse.ArgumentParser()
    # TO DO: parser.add_argument('model', type=str, nargs='+', help='BiLSTM_LRW, BiGRU_LRW, Conv3d_LRW') 
    parser.add_argument('--dataset_path', type=str, help='path to dataset')
    parser.add_argument('--feature_method', type=str, help='method used to collect facial features. ones of {dlib} (more to be implemented in future...)')
    # TO DO: parser.add_argument('--dataset_name', type=str, default='LRW', help='name of dataset. one of {LRW,LRS2}')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--dropout_p', type=float, default=0.3, help='dropout probabilty')
    # TO DO: parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--checkpoint_dir", type=str, help="checkpoints will be saved in this directory")
    parser.add_argument('--n_workers', type=int, default=1, help='number of workers to use during data generation')
    parser.add_argument('--feature_size', type=int, default=68*2, help='number of facial features used')
    parser.add_argument('--hidden_size', type=int, default=256, help='number of hidden units used')
    parser.add_argument('--num_layers', type=int, default=2, help='number of hidden layers used')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')

    args = parser.parse_args()
    
    wandb.init(project='cs269-lipreading', config={"model": 'BiLSTM-LRW'})
    wandb.config.update(args)

    options = wandb.config
    
    # Data loaders
    print('Setting up dataloaders...')
    train_set = LRWDataset_features(options["dataset_path"],'train',options['feature_method'])
    train_generator = DataLoader(train_set, batch_size=options["batch_size"], shuffle=True, num_workers=options["n_workers"])

    # Vocab dictionaries 
    vocab2int = train_set.vocab2int
    int2vocab = []
    vocab_size = 0
    for key in vocab2int.keys(): 
        int2vocab.append(key) 
        vocab_size += 1

    val_set = LRWDataset_features(options["dataset_path"],'val', options['feature_method'], vocab2int=vocab2int)
    val_generator = DataLoader(val_set, batch_size=options["batch_size"], num_workers=options["n_workers"])

    test_set = LRWDataset_features(options["dataset_path"],'test', options['feature_method'], vocab2int=vocab2int)
    test_generator = DataLoader(test_set, batch_size=options["batch_size"], num_workers=options["n_workers"])

    print('Dataloaders ready!')

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print(f'Using {device}')

    # Instantiate Model
    model = BiLSTM_LRW(options['feature_size'], 
                         vocab_size, options['hidden_size'], 
                         options['num_layers'], 
                         options['dropout_p'])
    model = model.to(device) # Assign CPU or GPU if available
    
    # Instantiate loss function and optimizer 
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options['learning_rate'], weight_decay=options["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Start training! 
    print('Starting training...')
    history = train_features_model(device, model, train_generator, val_generator, criterion, optimizer, scheduler, options)
    print('Training complete!')

    wandb.save("BiLSTM_LRW.h5")
