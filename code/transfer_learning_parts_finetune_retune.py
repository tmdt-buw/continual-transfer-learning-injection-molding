import numpy as np
import pandas as pd
from itertools import permutations
import copy

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import r2_score

from neural_network import Net


def load_and_transform_data(train_percentage, val_percentage, test_percentage):
    """
    Loads the data from the csv file.
    Splits the for each lego part into training, validation, and test set.
    Scales the input data and creates torch tensors.
    Stores the data into a dictionary.
    :return: A dictionary containing the processed training, validation, and test data for each part.
    """

    scale = StandardScaler()

    data_path = "../data/lego_data.csv"
    data_dict = {}
    parameters = np.array(['holding_pressure',
                           'holding_pressure_time',
                           'melt_temp',
                           'mold_temp',
                           'cooling_time',
                           'volume_flow'])
    quality_criteria = np.array(['max_deformation'])
    n_inputs = len(parameters)
    n_outputs = len(quality_criteria)

    data = pd.read_csv(data_path, sep=';')
    for lego_name in data['lego'].unique():
        mask = data['lego'] == lego_name
        lego_data = data[mask]
        data_dict[lego_name] = {}
        lego_data = lego_data[np.append(parameters, quality_criteria)]
        lego_data = shuffle(lego_data)

        # data is divided into training, validation and test sets
        data_n = len(lego_data)
        train_data = lego_data.values[0:int(data_n * train_percentage)]
        cval_data = lego_data.values[
                    int(data_n * train_percentage):int(data_n * (train_percentage + val_percentage))]
        test_data = lego_data.values[int(data_n * (train_percentage + val_percentage)):int(
            data_n * (train_percentage + val_percentage + test_percentage))]
        data_dict[lego_name]["train_x"] = torch.from_numpy(scale.fit_transform(train_data[:, 0:n_inputs])).float()
        data_dict[lego_name]["train_y"] = torch.from_numpy(train_data[:, n_inputs:n_inputs + n_outputs]).float()

        data_dict[lego_name]["val_x"] = torch.from_numpy(scale.transform(cval_data[:, 0:n_inputs])).float()
        data_dict[lego_name]["val_y"] = torch.from_numpy(cval_data[:, n_inputs:n_inputs + n_outputs]).float()

        data_dict[lego_name]["test_x"] = torch.from_numpy(scale.transform(test_data[:, 0:n_inputs])).float()
        data_dict[lego_name]["test_y"] = torch.from_numpy(test_data[:, n_inputs:n_inputs + n_outputs]).float()

    return data_dict



def train_model(model, optimizer, criterion, trainloader, valloader, num_epochs=2000, patience=50, logging=False, early_stopping=True):
    """
    Trains a pytorch model on training data.
    Performs early stopping on validation set. the data from the csv file.
    :return: A neural network model (pytorch) and the number of required epochs for training.
    """

    delta = 0.00001
    stop = False
    best_loss = None
    counter = 0

    for epoch in range(num_epochs):
        num_batches = 0
        batch_loss = 0.0
        val_loss = 0.0

        # train and compute training loss
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            model.train()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            num_batches += 1
        epoch_loss = batch_loss / num_batches

        # compute valiation loss
        num_batches = 0.0
        for batch_idx, (inputs, targets) in enumerate(valloader):
            model.eval()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            num_batches += 1
        val_loss = val_loss / num_batches

        # check early stopping on validation loss
        if early_stopping:
            if best_loss is None:
                best_loss = val_loss
            elif val_loss < best_loss - delta:
                best_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    stop = True

        if epoch % 100 == 0:
            if logging:
                print('Train Epoch: {} \tTrain_Loss: {:.6f} \t Validation_Loss: {:.6f}, Best_Loss: {:.6f}'.format(epoch, epoch_loss, val_loss, best_loss))

        if stop:
            if logging:
                print("EARLY STOPPING!")
            break

    return model, epoch



def evaluate_model(model, test_x, test_y):
    """
    Evaluates a model on a test set and computes the coefficient of determination
    :return: score of the model on the test set
    """
    model.eval()
    with torch.no_grad():
        test_x, test_y = test_x.to(device), test_y.to(device)
        net_out = model(test_x)
        score = r2_score(net_out.cpu().numpy(), test_y.cpu().numpy())
    return score



def transfer_retune_with_data_efficiency(data_dict, parts, list_id=0):
    """
    Iterates over all lego parts in parts list and performs continual transfer learning.
    A each iteration (part), iterates over data proportions, finetunes the model on new data and retunes on previous part data.
    Stores evaluation results in text files.
    """
    test_score_threshold = 0.95
    required_fracs = []     # required training data fractions to exceed threshold for each lego part in list
    required_epochs = []    # required training epochs for each lego part in list

    # iterate over all lego parts
    for idx, part in enumerate(parts):
        train_x = data_dict[part]['train_x']
        train_y = data_dict[part]['train_y']

        # if first part, train the neural network from scratch
        if idx == 0:
            print('Train on first lego part: {}'.format(part))
            fractions = np.arange(0, 1.01, 0.1)
            for frac in fractions:
                if frac == 0:
                    continue

                if frac > 0:
                    # load fraction of data and train network on it
                    train_x_incr = train_x[:int(frac * len(train_x))]
                    train_y_incr = train_y[:int(frac * len(train_y))]
                    train_dataset = TensorDataset(train_x_incr, train_y_incr)
                    trainloader = DataLoader(train_dataset, batch_size=10)

                    val_x = data_dict[part]['val_x']  # already torch tensors
                    val_y = data_dict[part]['val_y']
                    val_dataset = TensorDataset(val_x, val_y)
                    valloader = DataLoader(val_dataset, batch_size=10)

                    model = Net()
                    model.to(device)
                    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001, amsgrad=True)
                    criterion = nn.L1Loss()
                    model.train()
                    model, epochs = train_model(model, optimizer, criterion, trainloader, valloader, num_epochs=2000, logging=False)

                # test network on separate test set
                test_x = data_dict[part]['test_x']
                test_y = data_dict[part]['test_y']
                model.eval()
                test_score = evaluate_model(model, test_x, test_y)

                # when threshold exceeded, stop training for this part
                # store last two layers as part-related block
                if test_score >= test_score_threshold or frac == 1.0:
                    required_fracs.append(frac)
                    required_epochs.append(epochs)
                    model.part_layer_3[part] = copy.deepcopy(model.layers.fc3)
                    model.part_layer_4[part] = copy.deepcopy(model.layers.fc4)
                    break

        # if not first part, transfer model to new data by finetuning/retuning
        else:
            print('Transfer to part {}: {}'.format(idx+1, part))
            fractions = np.arange(0, 1.01, 0.1)
            bck_model = copy.deepcopy(model)
            for frac in fractions:
                if frac > 0:
                    # load fraction of data and train network on it (finetuning)
                    train_x_incr = train_x[:int(frac * len(train_x))]
                    train_y_incr = train_y[:int(frac * len(train_y))]

                    train_dataset = TensorDataset(train_x_incr, train_y_incr)
                    trainloader = DataLoader(train_dataset, batch_size=10)

                    val_x = data_dict[part]['val_x']  # already torch tensors
                    val_y = data_dict[part]['val_y']
                    val_dataset = TensorDataset(val_x, val_y)
                    valloader = DataLoader(val_dataset, batch_size=10)

                    model = copy.deepcopy(bck_model)
                    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001, amsgrad=True)
                    model, epochs = train_model(model, optimizer, criterion, trainloader, valloader, num_epochs=2000, logging=False)

                # test network on separate test set for new part
                test_x = data_dict[part]['test_x']
                test_y = data_dict[part]['test_y']
                test_score = evaluate_model(model, test_x, test_y)

                # when threshold exceeded, stop training for this part
                # store last two layers as part-related block
                # retune other part-related blocks for previous parts
                if test_score >= test_score_threshold or frac == 1.0:
                    required_fracs.append(frac)
                    required_epochs.append(epochs)
                    model.part_layer_3[part] = copy.deepcopy(model.layers.fc3)
                    model.part_layer_4[part] = copy.deepcopy(model.layers.fc4)

                    # Freeze first two layers
                    model.layers.fc1.weight.requires_grad = False
                    model.layers.fc1.bias.requires_grad = False
                    model.layers.fc2.weight.requires_grad = False
                    model.layers.fc2.bias.requires_grad = False

                    # iterate over previous parts and retune
                    for prev_id in np.arange(0, idx):
                        prev_part = parts[prev_id]
                        print('Retuning on previous part {}: {}'.format(prev_id + 1, prev_part))
                        fracs_used = required_fracs[prev_id]

                        # load train and validation data for previous part
                        prev_train_x = data_dict[prev_part]['train_x']
                        prev_train_y = data_dict[prev_part]['train_y']
                        prev_train_x = prev_train_x[:int(fracs_used * len(prev_train_x))]
                        prev_train_y = prev_train_y[:int(fracs_used * len(prev_train_y))]
                        train_dataset = TensorDataset(prev_train_x, prev_train_y)
                        trainloader = DataLoader(train_dataset, batch_size=10)
                        prev_val_x = data_dict[prev_part]['val_x']
                        prev_val_y = data_dict[prev_part]['val_y']
                        val_dataset = TensorDataset(prev_val_x, prev_val_y)
                        valloader = DataLoader(val_dataset, batch_size=10)

                        # Switch last two layers to corresponding part-related block and retrain network
                        model.layers.fc3 = copy.deepcopy(model.part_layer_3[prev_part])
                        model.layers.fc4 = copy.deepcopy(model.part_layer_4[prev_part])
                        optimizer = optim.Adam(model.parameters(), lr=0.01)
                        model, epochs = train_model(model, optimizer, criterion, trainloader, valloader, num_epochs=2000, early_stopping=True, logging=False)
                        model.part_layer_3[prev_part] = copy.deepcopy(model.layers.fc3)
                        model.part_layer_4[prev_part] = copy.deepcopy(model.layers.fc4)


                    # Replace last two layers again with current part-related block
                    model.layers.fc3 = copy.deepcopy(model.part_layer_3[part])
                    model.layers.fc4 = copy.deepcopy(model.part_layer_4[part])

                    # Unfreeze first two layers for finetuning of next part
                    model.layers.fc1.weight.requires_grad = True
                    model.layers.fc1.bias.requires_grad = True
                    model.layers.fc2.weight.requires_grad = True
                    model.layers.fc2.bias.requires_grad = True
                    break

    # After training of all parts, extract final scores (coefficient of determination) for every part
    test_scores_final = []
    for idx, part in enumerate(parts):
        # Switch layer fc3 to corresponding layer for previous part
        model.layers.fc3 = copy.deepcopy(model.part_layer_3[part])
        model.layers.fc4 = copy.deepcopy(model.part_layer_4[part])
        test_x = data_dict[part]['test_x']  # already torch tensors
        test_y = data_dict[part]['test_y']
        test_score = evaluate_model(model, test_x, test_y)
        test_scores_final.append(test_score)


    print('Required_Fractions: {}'.format(required_fracs))
    print('Required_Epochs: {}'.format(required_epochs))
    print('Test Scores Final: {}'.format(test_scores_final))

    np.savetxt('../results/finetune_retune_fracs_perm_{}_seed_{}.txt'.format(list_id, seed), required_fracs, newline=" ")
    np.savetxt('../results/finetune_retune_epochs_perm_{}_seed_{}.txt'.format(list_id, seed), required_epochs,
               newline=" ")
    np.savetxt('../results/finetune_retune_scores_final_perm_{}_seed_{}.txt'.format(list_id, seed), test_scores_final,
               newline=" ")




if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_percentage = 0.75
    val_percentage = 0.1
    test_percentage = 0.15

    # load the data
    data_dict = load_and_transform_data(train_percentage, val_percentage, test_percentage)
    part_list = ['6x1_Lego', '4x2_Lego', '3x2_Lego', '3x1_Lego', '6x2_Lego', '4x1_Lego']

    # conduct transfer learning experiments 5 times for a single part list
    # for the paper, evaluation was conducted on all 720 permutations of the list with 10 seeds per list
    for seed in np.arange(5):
        print('*'*40)
        print('Transfer Learning for parts {} with seed {}'.format(part_list, seed))
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        transfer_retune_with_data_efficiency(data_dict, np.array(part_list))
