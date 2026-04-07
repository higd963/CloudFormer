import argparse
from netCDF4 import Dataset
from os import listdir
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.nn as nn
from tqdm import tqdm
import gzip
import pickle
from model import CloudFractionTransformer

def model_train(model_type, model_save, date, savefile, input_vars=['qci','rh','tt','level']):
    adir = '../../../../data/DATABASE/REANALYSIS/ERA5/1deg_6hr/' + date + '/'
    filelist = sorted([s for s in listdir(adir) if "prs" in s])

    qci_list, rh_list, tt_list, qs_list, level_list, cf_list = [], [], [], [], [], []
    qc_list, qi_list = [], []

    for fname in filelist:
        print(fname)
        with Dataset(adir + fname, "r") as otype:
            qi = np.array(otype.variables['ciwc'])
            qc = np.array(otype.variables['clwc'])
            rh = np.array(otype.variables['r']) * 0.01
            tt = np.array(otype.variables['t'])
            qs = np.array(otype.variables['cswc'])
            cf = np.array(otype.variables['cc'])
            level = np.array(otype.variables['level'])
            qci = (qc + qi) * 1000.0
            qc = qc * 1000.0
            qi = qi * 1000.0
            qs = qs * 1000.0

            for t in range(cf.shape[0]):
                for lat in range(cf.shape[2]):
                    for lon in range(cf.shape[3]):
                        qci_list.append(qci[t,:,lat,lon])
                        qc_list.append(qc[t,:,lat,lon])
                        qi_list.append(qi[t,:,lat,lon])
                        rh_list.append(rh[t,:,lat,lon])
                        tt_list.append(tt[t,:,lat,lon])
                        qs_list.append(qs[t,:,lat,lon])
                        cf_list.append(cf[t,:,lat,lon])
                        level_list.append(level)

    input_dict = {
        'qci': np.array(qci_list), 'qc': np.array(qc_list), 'qi': np.array(qi_list),
        'rh': np.array(rh_list), 'tt': np.array(tt_list), 'qs': np.array(qs_list),
        'level': np.array(level_list)  # pressure
    }

    norm_input_dict = {}
    for k in input_dict:
        if k == 'level':
            norm_input_dict[k] = input_dict[k] 
        else:
            minval = input_dict[k].min()
            maxval = input_dict[k].max()
            norm_input_dict[k] = (input_dict[k] - minval) / (maxval - minval + 1e-8)

    input_stack = [norm_input_dict[var][:,10:] for var in input_vars]
    input_data = np.stack(input_stack, axis=1)  # (N, 4, L)
    output_data = np.stack([np.array(cf_list)[:,10:]], axis=1)  # (N, 1, L)

    dataset = TensorDataset(torch.Tensor(input_data), torch.Tensor(output_data))
    lengths = [round(len(dataset)*0.7), round(len(dataset)*0.3)]
    if sum(lengths) != len(dataset):
        lengths[0] += len(dataset) - sum(lengths)
    train_set, valid_set = random_split(dataset, lengths)
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1024, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    input_dim = len([v for v in input_vars if v != 'level'])


    if model_save:
        var_suffix = "_" + "_".join(input_vars)
        model = CloudFractionTransformer(input_dim=input_dim).to(device)
        state_dict = torch.load(f'../03_results/00_{model_type}/var/{savefile}{var_suffix}.pt')
        model.load_state_dict(state_dict)

    else:
        model = CloudFractionTransformer(input_dim=input_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience = 10
    no_improve = 0

    for epoch in range(50):
        print(f"[Epoch {epoch+1}]")
        model.train()
        total_loss = 0
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.squeeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target.squeeze(1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(valid_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            var_suffix = "_" + "_".join(input_vars)
            torch.save(model.state_dict(), f'../03_results/00_{model_type}/var/{savefile}{var_suffix}.pt')
            print("Model saved.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break


def model_test(model_type, model_save, date, savefile, input_vars=['qci','rh','tt','level']):
    adir = '../../../../data/DATABASE/REANALYSIS/ERA5/1deg_6hr/' + date + '/'
    filelist = sorted([s for s in listdir(adir) if "prs" in s])

    qci_list, rh_list, tt_list, qs_list, level_list, cf_list = [], [], [], [], [], []
    qc_list, qi_list = [], []

    for fname in filelist:
        print(fname)
        with Dataset(adir + fname, "r") as otype:
            qi = np.array(otype.variables['ciwc'])
            qc = np.array(otype.variables['clwc'])
            rh = np.array(otype.variables['r']) * 0.01
            tt = np.array(otype.variables['t'])
            qs = np.array(otype.variables['cswc'])
            cf = np.array(otype.variables['cc'])
            level = np.array(otype.variables['level'])
            qci = (qc + qi) * 1000.0
            qc = qc * 1000.0
            qi = qi * 1000.0
            qs = qs * 1000.0

            for t in range(cf.shape[0]):
                for lat in range(cf.shape[2]):
                    for lon in range(cf.shape[3]):
                        qci_list.append(qci[t,:,lat,lon])
                        qc_list.append(qc[t,:,lat,lon])
                        qi_list.append(qi[t,:,lat,lon])
                        rh_list.append(rh[t,:,lat,lon])
                        tt_list.append(tt[t,:,lat,lon])
                        qs_list.append(qs[t,:,lat,lon])
                        cf_list.append(cf[t,:,lat,lon])
                        level_list.append(level)

    input_dict = {
        'qci': np.array(qci_list), 'qc': np.array(qc_list), 'qi': np.array(qi_list),
        'rh': np.array(rh_list), 'tt': np.array(tt_list), 'qs': np.array(qs_list),
        'level': np.array(level_list)
    }

    norm_input_dict = {}
    for k in input_dict:
        if k == 'level':
            norm_input_dict[k] = input_dict[k]
        else:
            minval = input_dict[k].min()
            maxval = input_dict[k].max()
            norm_input_dict[k] = (input_dict[k] - minval) / (maxval - minval + 1e-8)

    input_stack = [norm_input_dict[var][:,10:] for var in input_vars]
    input_data = np.stack(input_stack, axis=1)
    output_data = np.stack([np.array(cf_list)[:,10:]], axis=1)

    test_data = TensorDataset(torch.Tensor(input_data), torch.Tensor(output_data))
    test_loader = DataLoader(test_data, batch_size=500, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    var_suffix = "_" + "_".join(input_vars)
    input_dim = len([v for v in input_vars if v != 'level'])
    model = CloudFractionTransformer(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(f'../03_results/00_{model_type}/var/{savefile}{var_suffix}.pt'))

    model.eval()
    preds, trues, losses = [], [], []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target.squeeze(1))
            preds.append(output.cpu().numpy())
            trues.append(target.squeeze(1).cpu().numpy())
            losses.append(loss.item())

    print(f"Test MSE: {np.mean(losses):.4f}")

    pred_all = np.concatenate(preds, axis=0)
    true_all = np.concatenate(trues, axis=0)

    with gzip.open(f'../03_results/00_{model_type}/var/pred_{savefile}{var_suffix}_{date}.pickle', 'wb') as f:
        pickle.dump(pred_all, f)
    with gzip.open(f'../03_results/00_{model_type}/var/true_{savefile}{var_suffix}_{date}.pickle', 'wb') as f:
        pickle.dump(true_all, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save', type=str)
    parser.add_argument('--date', type=str)
    parser.add_argument('--savefile', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--input_vars', nargs='+')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    args = parser.parse_args()

    model_save = args.model_save.lower() == 'true'
    date = args.date
    savefile = args.savefile
    model_type = args.model_type
    input_vars = args.input_vars if args.input_vars else ['qci','rh','tt','level']
    mode = args.mode

    if mode == 'train':
        model_train(model_type, model_save, date, savefile, input_vars)
    else:
        model_test(model_type, model_save, date, savefile, input_vars)
