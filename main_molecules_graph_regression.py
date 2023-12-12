import numpy as np
import os
import time
import random
import argparse, json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from dict_processing import dict_processing
from train.train_co_graph_regression import train_epoch, evaluate_network
from sklearn.model_selection import KFold
from nets.co_graph_regression.load_net import gnn_model
from data.data import LoadData

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def gpu_setup(use_gpu, gpu_id):
    # 设置CUDA环境以使用GPU（如果可用），并返回相应的设备对象。
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

def view_model_param(MODEL_NAME, net_params):
    # 加载GNN模型并打印模型参数的详细信息，包括每个参数的数据类型和总参数数量。
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
        print("Parameter data type:", param.dtype)
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    # 这是代码的核心部分，它执行模型的训练和验证过程。它使用K折交叉验证来评估模型的性能，并记录训练和验证过程中的各种统计信息。
    DATASET_NAME = dataset.name
    t0 = time.time()
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))

    # 设定日志记录器
    writer = SummaryWriter(log_dir = log_dir)
    # 设置随机种子以保证结果的可重复性
    random.seed(params['seed_net'])
    np.random.seed(params['seed_net'])
    torch.manual_seed(params['seed_net'])
    if device.type == 'cuda':
       torch.cuda.manual_seed(params['seed_net'])

    # 使用K折交叉验证来分割数据集
    n_folds = 5
    data = dataset.data
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=params['cv_seed'])

    # import train and evaluate functions
    cv_scores = []
    # 保存表现最好的模型。
    for fold, (train_idx, test_idx) in enumerate(kfold.split(data.graph_lists, data.graph_labels)):
    # 在每一折中，进行以下操作：
        per_epoch_time = []
        # 初始化模型和优化器
        model = gnn_model(MODEL_NAME, net_params)
        model = model.to(device)
        epoch_train_losses, epoch_test_losses = [], []
        epoch_train_MAEs, epoch_train_MSEs, epoch_test_MAEs, epoch_test_MSEs = [], [], [], []
        print(f'Fold {fold + 1}')
        # t.set_description('Fold %d' % fold + 1))
        train_set = [data[i] for i in train_idx]
        test_set = [data[i] for i in test_idx]
        train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
        test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
        optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
        if net_params['scheduler_ReduceLROnPlateau']:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=params['lr_reduce_factor'],
                                                         patience=params['lr_schedule_patience'],
                                                         verbose=True)
        if net_params['scheduler_step']:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        if net_params['scheduler_onecycle']:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=params['init_lr'],
                epochs=params['epochs'],
                steps_per_epoch=len(train_loader),
                # pct_start=pct_start,
                pct_start=0.3,
            )
        best_test_mae = 99.99
        best_epoch = 0
        # 进行多个epoch的训练和验证
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                model.train()
                epoch_train_loss, epoch_train_mae, epoch_train_mse, optimizer, scheduler = train_epoch(model, optimizer, device, train_loader, scheduler)
                model.eval()
                epoch_test_loss, epoch_test_mae, epoch_test_mse = evaluate_network(model, device, test_loader)
                if epoch_test_mae < best_test_mae:
                    best_test_mae = epoch_test_mae
                    best_epoch = epoch
                    train_mae = epoch_train_mae
                    train_mse = epoch_train_mse
                    test_mae = epoch_test_mae
                    test_mse = epoch_test_mse

                epoch_train_losses.append(epoch_train_loss)
                epoch_test_losses.append(epoch_test_loss)
                epoch_train_MAEs.append(epoch_train_mae)
                epoch_train_MSEs.append(epoch_train_mse)
                epoch_test_MAEs.append(epoch_test_mae)
                epoch_test_MSEs.append(epoch_test_mse)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('test/_loss', epoch_test_loss, epoch)
                writer.add_scalar('train/_mae', epoch_train_mae, epoch)
                writer.add_scalar('train/_mse', epoch_train_mse, epoch)
                writer.add_scalar('test/_mae', epoch_test_mae, epoch)
                writer.add_scalar('test/_mse', epoch_test_mse, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, test_loss=epoch_test_loss, train_MAE=epoch_train_mae,
                              test_MAE=epoch_test_mae,train_MSE=epoch_train_mse, test_MSE=epoch_test_mse)

                per_epoch_time.append(time.time() - start)
                # optimizer.step()
                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if (time.time()-t0) > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
        # 记录每个epoch的损失、平均绝对误差（MAE）、均方误差（MSE）
        cv_scores.append(best_test_mae)
        print("Best epoch: {:.4f}".format(best_epoch))
        print("Best test MAE: {:.4f}".format(best_test_mae))
        print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
        print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

        writer.close()
        if fold==0:
            with open(write_file_name + '.txt', 'w') as f:
                f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
            FINAL RESULTS\nfold:{}\nTEST MAE: {:.4f}\nTRAIN MAE: {:.4f}\nTEST MSE: {:.4f}\nTRAIN MSE: {:.4f}\n
            Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
                  .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'], fold,
                          test_mae, train_mae, test_mse, train_mse, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
        else:
            with open(write_file_name + '.txt', 'a') as f:
                f.write("""fold:{}\nTEST MAE: {:.4f}\nTRAIN MAE: {:.4f}\nTEST MSE: {:.4f}\nTRAIN MSE: {:.4f}\n
                Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n""" \
                        .format(fold,test_mae, train_mae, test_mse, train_mse, epoch, (time.time() - t0) / 3600,
                                np.mean(per_epoch_time)))
        np.savetxt(write_file_name+ str(fold) +'epoch_train_MAEs.txt',epoch_train_MAEs)
        np.savetxt(write_file_name+ str(fold)+'epoch_train_MSEs.txt',epoch_train_MSEs)
        np.savetxt(write_file_name+ str(fold)+'epoch_test_MAEs.txt',epoch_test_MAEs)
        np.savetxt(write_file_name+ str(fold)+'epoch_test_MSEs.txt',epoch_test_MSEs)
        torch.save(model.state_dict(), root_ckpt_dir+ 'model_params'+ str(fold) + '.pth')
    mean_mae = np.mean(cv_scores)
    with open(write_file_name + '.txt', 'a') as f:
        f.write("""mean_mae:{}""" \
                .format(mean_mae))
    t.set_postfix(meanmae=mean_mae)
    print(f'Mean cross-validated mean_mae: {mean_mae}')


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with param details")  # 用到了
    args = parser.parse_args()
    # 根据提供的配置文件加载配置
    args.config = 'configs/catalyze_graph_regression_CO_init.json'
    with open(args.config) as f:
        config = json.load(f)

    # 设置CUDA设备
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])

    # 从配置文件中读取并设置训练参数和网络参数
    params = config['params']

    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']

    # model, dataset, out_dir
    MODEL_NAME = config['model']  # "GraphTransformer"

    # 加载数据集
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME, params)
    out_dir = config['out_dir']
    net_params['num_atom_type'] = dataset.num_atom_type
    net_params['num_bond_type'] = dataset.num_bond_type
    net_params['num_theta_type'] = dataset.num_theta_type

    # 创建所需的目录结构来保存日志、检查点等
    root_log_dir = out_dir + str(params['seed'])+'/'+'logs/' + MODEL_NAME + "_soap" + str(net_params['h_soap'])
    root_ckpt_dir = out_dir + str(params['seed'])+'/' + 'checkpoints/' + MODEL_NAME + "_soap" + str(net_params['h_soap'])
    write_file_name = out_dir + str(params['seed'])+'/'+ 'results/' + MODEL_NAME + "_soap" + str(net_params['h_soap'])
    write_config_file = out_dir + str(params['seed']) +'/'+ 'configs/' + MODEL_NAME + "_soap" + str(net_params['h_soap'])
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(root_log_dir):
        os.makedirs(root_log_dir)
    if not os.path.exists(root_ckpt_dir):
        os.makedirs(root_ckpt_dir)
    if not os.path.exists(write_file_name):
        os.makedirs(write_file_name)
    if not os.path.exists(write_config_file):
        os.makedirs(write_config_file)
    if net_params['h_soap']:
        net_params['soap_enc_dim'] = dataset.soap_dim

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)

    # 调用train_val_pipeline函数来开始训练和验证流程
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)  # model_name = 'GraphTransformer';

main()