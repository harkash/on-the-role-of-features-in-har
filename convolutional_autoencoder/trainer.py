import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

# ---------------------------------------------------------------------------------------------------------------------


def train_ae(model, data_loaders, num_epochs, dataset_sizes, args):
    best_model_wts = copy.deepcopy(model.state_dict())

    # Setting the criterion and optimizer+scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.8)

    for epoch in range(num_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            actual_labels = []

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = Variable(inputs).float()
                labels = Variable(labels).type(torch.LongTensor)

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, inputs)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                actual_labels.extend(labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            # deep copy the model
            # if phase == 'train' and epoch_loss <= best_loss:
            #     best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def perform_inference(model, data_loaders, dataset_sizes, batch_size, hidden_size, perform_norm):
    print('Performing the inference on the model')
    model.eval()  # Since we do only inference

    data = {}
    since = time.time()
    for phase in ['train', 'val', 'test']:
        encoded_data = np.zeros((dataset_sizes[phase], hidden_size))
        all_labels = np.zeros((dataset_sizes[phase],))
        count = 0

        for inputs, labels in data_loaders[phase]:
            inputs = Variable(inputs).float()
            labels = Variable(labels).type(torch.LongTensor)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            _, hidden = model(inputs)
            encoded_data[count:count + batch_size, :] = hidden.cpu().data.numpy()
            all_labels[count:count + batch_size] = labels.cpu().data.numpy()
            count += batch_size

        # Saving the data into .npy files
        data[phase + '_data'] = encoded_data
        data[phase + '_labels'] = all_labels

    time_taken = time.time() - since
    print('The time taken to compute all features is {} seconds'.format(time_taken))
    print('The average time for computing a representation is {} seconds'.format(time_taken / (data['train_data'].shape[0] + data['val_data'].shape[0] + data['test_data'].shape[0])))

    for phase in ['train', 'val', 'test']:
        print('The shape of {} data is {}, and train labels is {}'.format(phase, data[phase + '_data'].shape,
                                                                          data[phase + '_labels'].shape))

    # Perform normalization on the features
    if perform_norm:
        scaler = StandardScaler()
        scaler.fit(data['train_data'])
        data['train_data'] = scaler.transform(data['train_data'])
        data['val_data'] = scaler.transform(data['val_data'])
        data['test_data'] = scaler.transform(data['test_data'])
        print('Normalization complete!')

    return data


def train_mlp(model, data_loaders, criterion, optimizer, scheduler, num_epochs, dataset_sizes):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1_score = 0.0

    for epoch in range(num_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            actual_labels = []
            pred_labels = []

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = Variable(inputs).float()
                labels = Variable(labels).type(torch.LongTensor)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                actual_labels.extend(labels.cpu().data.numpy())
                pred_labels.extend(preds.cpu().data.numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_f_score_weighted = f1_score(actual_labels, pred_labels, average='weighted')
            epoch_f_score_macro = f1_score(actual_labels, pred_labels, average='macro')

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} F1 score weighted: {:.4f}, F1 score macro: {:.4f}'.format(phase, epoch_f_score_weighted,
                                                                                epoch_f_score_macro))
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc and epoch > 0.5*num_epochs:
                best_f1_score = epoch_f_score_macro
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print('Best val f1-score: {:.4f}, accuracy: {:.4f}'.format(best_f1_score, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
