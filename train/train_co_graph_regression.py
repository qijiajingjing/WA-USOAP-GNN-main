"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
from train.metrics import MAE
from train.metrics import MSE

def train_epoch(model, optimizer, device, data_loader, scheduler):
    # model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    epoch_train_mse = 0
    nb_data = 0
    # print(len(data_loader))
    for iter, (batch_graphs, l_batch_graphs, batch_targets) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        l_batch_graphs = l_batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        _ = batch_graphs.ndata['_'].to(device)
        batch_x_soap = batch_graphs.ndata['soap_enc'].to(device)
        l_batch_x = l_batch_graphs.ndata['feat'].to(device)
        l_batch_e = l_batch_graphs.edata['feat'].to(device)
        batch_targets = batch_targets.to(device).float()
        # print(batch_targets)
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, l_batch_graphs, batch_x, batch_e, l_batch_x, l_batch_e, _, batch_x_soap,)
        # print(batch_scores)
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        epoch_train_mse += MSE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    epoch_train_mse /= (iter + 1)
    return epoch_loss, epoch_train_mae, epoch_train_mse, optimizer, scheduler

def evaluate_network(model, device, data_loader, show = False):
    # model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    epoch_test_mse = 0
    nb_data = 0
    scores, targets = [],[]
    with torch.no_grad():
        for iter, (batch_graphs, l_batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            l_batch_graphs = l_batch_graphs.to(device)
            l_batch_x = l_batch_graphs.ndata['feat'].to(device)
            l_batch_e = l_batch_graphs.edata['feat'].to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            _ = batch_graphs.ndata['_'].to(device)
            batch_targets = batch_targets.to(device)
            batch_x_soap = batch_graphs.ndata['soap_enc'].to(device)
            batch_scores = model.forward(batch_graphs, l_batch_graphs, batch_x, batch_e, l_batch_x, l_batch_e, _, batch_x_soap)
            loss = model.loss(batch_scores, batch_targets)
            scores.extend(batch_scores.cpu().numpy())
            targets.extend(batch_targets.cpu().numpy())
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            epoch_test_mse += MSE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        epoch_test_mse /= (iter + 1)
        if show == True:
            return epoch_test_loss, epoch_test_mae, epoch_test_mse, targets, scores
    return epoch_test_loss, epoch_test_mae, epoch_test_mse
