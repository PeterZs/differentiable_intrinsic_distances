from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import os
import json
import visdom
import time
from chamfer_python import *
from dataset import *
from model import *
from utils import *
from utils_parallel import *

if __name__ == '__main__':  # OH: Wrapping the main code with __main__ check is necessary for Windows compatibility
    # of the multi-process data loader (see pytorch documentation)

    # =============PARAMETERS======================================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--nepoch', type=int, default=100000, help='number of epochs to train for')
    parser.add_argument('--model', type=str, default='', help='optional reload model path')
    parser.add_argument('--save_path', type=str, default='optimization', help='save path')
    parser.add_argument('--env', type=str, default="3DCODED_supervised", help='visdom environment')  # OH: TODO edit

    opt = parser.parse_args()
    print(opt)

    # =============Loading Intrinsic Properties of GT Completion======================================== #
    x = sio.loadmat("D:/shape_completion/data/eigendecomposition/downsampled_tr_reg_004.mat")
    adj_VF = torch.from_numpy(x['adj_VF']).cuda().float()
    num_vertices = adj_VF.shape[0]
    num_triangles = adj_VF.shape[1]
    num_evec = 10
    LBO_gt = torch.from_numpy(x['L']).cuda().double()
    LBO_gt = LBO_gt.unsqueeze(0).expand(opt.batchSize, num_vertices, num_vertices)
    triv = torch.from_numpy(x['F'].astype(int)).cuda().unsqueeze(0).expand(opt.batchSize, num_triangles, 3)
    triv = triv - 1  # zero based index
    vertices = torch.from_numpy(x['V']).cuda().unsqueeze(0).expand(opt.batchSize, num_vertices, 3)

    phi = torch.from_numpy(x['Phi']).cuda().double()
    phi = phi[:,:num_evec].unsqueeze(0).expand(opt.batchSize, num_vertices, num_evec)
    evals = torch.from_numpy(x['Lambda']).cuda().double()
    evals = evals[:num_evec].unsqueeze(0).expand(opt.batchSize, num_evec, 1).transpose(2, 1)
    area_V = torch.from_numpy(x['A']).cuda().double().unsqueeze(0).expand(opt.batchSize, num_vertices, 1)

    # =============DEFINE stuff for logs ======================================== #
    # Launch visdom for visualization
    vis = visdom.Visdom(port=8888, env=opt.env)
    save_path = opt.save_path
    dir_name = os.path.join('./log/', save_path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    logname = os.path.join(dir_name, 'log.txt')

    blue = lambda x: '\033[94m' + x + '\033[0m'

    opt.manualSeed = 1  # random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    L2curve_train_smpl = []
    L2curve_val_smlp = []

    # meters to record stats on learning
    train_loss_L2_smpl = AverageValueMeter()
    val_loss_L2_smpl = AverageValueMeter()
    tmp_val_loss = AverageValueMeter()

    # ===================CREATE optimizer================================= #
    lrate = 0.01 # learning rate
    pointsReconstructed = torch.randn((2, num_vertices, 3), requires_grad=True,device="cuda", dtype = torch.float)
    optimizer = optim.Adam([{'params' : pointsReconstructed}], lr=lrate)

    #with open(logname, 'a') as f:  # open and append
    #    f.write(str(network) + '\n')
    # ========================================================== #

    # =============start of the learning loop ======================================== #
    for epoch in range(opt.nepoch):
        if (epoch % 100) == 99:
            lrate = lrate/2.0
            optimizer = optim.Adam([{'params': pointsReconstructed}], lr=lrate)

        # TRAIN MODE
        train_loss_L2_smpl.reset()
        optimizer.zero_grad()
        gt = vertices

        # OH: place on GPU
        gt = gt.cuda().float()
        if epoch == 0:
            pointsReconstructed.data = gt[:,:,:3].data.clone() # initialization
            pointsReconstructed.data = pointsReconstructed.data + 0.05*torch.randn_like(pointsReconstructed.data)

        #D_reconstructed = calc_euclidean_dist_matrix(pointsReconstructed[0,:,:].unsqueeze(0)).float()
        #D_reconstructed = D_reconstructed.expand(opt.batchSize, num_vertices, num_vertices)
        #D_gt = calc_euclidean_dist_matrix(gt[0,:,:3].unsqueeze(0)).float()
        #D_gt = D_gt.expand(opt.batchSize, num_vertices, num_vertices)
        #loss_euclidean = torch.mean((D_reconstructed - D_gt) ** 2)

        L_reconstructed, area_matrix, area_matrix_inv, W = LBO(pointsReconstructed.double(), triv.long())
        L_sym = torch.bmm(area_matrix ** 0.5, torch.bmm(L_reconstructed, area_matrix_inv ** 0.5))
        K = torch.tensor(num_evec)
        N = torch.tensor(num_vertices)
        phi_reconstructed, lambda_reconstructed = EigendecompositionParallel(L_sym[0].cpu().double(), K, N)
        phi_reconstructed = phi_reconstructed.unsqueeze(0).expand(opt.batchSize, num_vertices, num_evec).cuda()
        phi_reconstructed = torch.bmm(area_matrix_inv ** 0.5, phi_reconstructed) #convert frfom eigenfunctions of L_sym to L
        square_norm_phi_reconstructed = torch.bmm(phi_reconstructed.transpose(2,1), torch.bmm(area_matrix, phi_reconstructed))
        square_norm_phi_reconstructed = torch.diag(square_norm_phi_reconstructed[0,:,:])
        square_norm_phi_reconstructed = square_norm_phi_reconstructed.unsqueeze(0).unsqueeze(1)
        phi_reconstructed = phi_reconstructed/(square_norm_phi_reconstructed ** 0.5)
# =============debug phi calculation ======================================== #
        #lambda_true = evals[0,0,:].cpu().data.numpy()
        #lambda_rec = lambda_reconstructed.cpu().data.numpy()

        #print(lambda_true.shape)
        #print(lambda_rec.shape)
        #vis.line(X=np.column_stack((np.arange(lambda_true.shape[0]), np.arange(lambda_reconstructed.shape[0]))),
        #         Y=np.column_stack((lambda_true, lambda_reconstructed)),
        #         win='lambda',
        #         opts=dict(title="lambda", legend=["lambda_true" + opt.env, "lambda_rec" + opt.env, ]))

        #phi_true = np.transpose(phi[0,:,0].cpu().data.numpy())
        #phi_rec = np.transpose(phi_reconstructed[0,:,0].cpu().data.numpy())
        #vis.line(X=np.column_stack((np.arange(phi_rec.shape[0]), np.arange(phi_true.shape[0]))),
        #         Y=np.column_stack((phi_true, phi_rec)),
        #         win='phi 0',
        #         opts=dict(title="phi 0", legend=["phi_true" + opt.env, "phi_rec" + opt.env, ]))

        #phi_true = np.transpose(phi[0,:,1].cpu().data.numpy())
        #phi_rec = np.transpose(phi_reconstructed[0,:,1].cpu().data.numpy())
        #vis.line(X=np.column_stack((np.arange(phi_rec.shape[0]), np.arange(phi_true.shape[0]))),
        #         Y=np.column_stack((phi_true, phi_rec)),
        #         win='phi 1',
        #         opts=dict(title="phi 1", legend=["phi_true" + opt.env, "phi_rec" + opt.env, ]))

        #phi_true = np.transpose(phi[0,:,2].cpu().data.numpy())
        #phi_rec = np.transpose(phi_reconstructed[0,:,2].cpu().data.numpy())
        #vis.line(X=np.column_stack((np.arange(phi_rec.shape[0]), np.arange(phi_true.shape[0]))),
        #         Y=np.column_stack((phi_true, phi_rec)),
        #         win='phi 2',
        #         opts=dict(title="phi 2", legend=["phi_true" + opt.env, "phi_rec" + opt.env, ]))

        #phi_true = np.transpose(phi[0,:,3].cpu().data.numpy())
        #phi_rec = np.transpose(phi_reconstructed[0,:,3].cpu().data.numpy())
        #vis.line(X=np.column_stack((np.arange(phi_rec.shape[0]), np.arange(phi_true.shape[0]))),
        #         Y=np.column_stack((phi_true, phi_rec)),
        #         win='phi 3',
        #         opts=dict(title="phi 3", legend=["phi_true" + opt.env, "phi_rec" + opt.env, ]))
# ====================================================================================================== #



        D_diff_rec = calc_euclidean_dist_matrix(phi_reconstructed)
        D_diff_gt = calc_euclidean_dist_matrix(phi)
        loss_diff = torch.mean((D_diff_rec - D_diff_gt) ** 2)
        #loss_phi = torch.mean((phi_reconstructed - phi) ** 2)
        #loss_L = torch.mean((L_reconstructed - LBO_gt) ** 2)
        #v.backward(torch.ones(num_vertices,num_evec).double())


        #d_chamfer = torch.mean(distChamfer(gt[:,:,:3],pointsReconstructed) ** 2)

        loss_net = loss_diff
        loss_net.backward()
        train_loss_L2_smpl.update(loss_net.item())
        optimizer.step()  # gradient update


        # VIZUALIZE
        vis.scatter(X=pointsReconstructed[0,:,:3].data.cpu(), win='Train_output',
                    opts=dict(title="Train_output",markersize=2, ), )
        vis.scatter(X=gt[0,:,:3].data.cpu(), win='Train_Ground_Truth',
                    opts=dict(title="Train_Ground_Truth", markersize=2, ), )
        save_model = pointsReconstructed.cpu().detach().numpy()
        sio.savemat('optimization_result.mat',
                    {'pointsReconstructed': save_model})

        print('[%d: %d] train loss:  %f' % (epoch, loss_net,  loss_net.item()))

        # UPDATE CURVES
        L2curve_train_smpl.append(train_loss_L2_smpl.avg)

        vis.line(X=np.arange(len(L2curve_train_smpl)),
                 Y=np.array(L2curve_train_smpl),
                 win='loss',
                 opts=dict(title="loss", legend=["L2curve_train_smpl"]))

        vis.line(X=np.arange(len(L2curve_train_smpl)),
                 Y=np.log(np.array(L2curve_train_smpl)),
                 win='log',
                 opts=dict(title="log", legend=["L2curve_train_smpl" + opt.env]))


