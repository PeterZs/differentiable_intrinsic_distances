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
from utils import *
from chamfer_python import *

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--num_evec', type=int, default=10, help='visdom environment')
parser.add_argument('--nepoch', type=int, default=100000, help='number of epochs to train for')
parser.add_argument('--full_shape_path', type=str, default="./../data/eigendecomposition/downsampled_tr_reg_000.mat", help='save path')
parser.add_argument('--partial_shape_path', type=str, default="./../data/faust_range_scans/dataset/tr_reg_004_001.mat", help='save path')
parser.add_argument('--save_path', type=str, default='optimization', help='save path')
parser.add_argument('--env', type=str, default="diffusion", help='visdom environment')
parser.add_argument('--batchSize', type=int, default=2, help='save path')

opt = parser.parse_args()
print(opt)

# =============Loading Intrinsic Properties of GT Completion======================================== #
x = sio.loadmat(opt.full_shape_path)
adj_VF = torch.from_numpy(x['adj_VF']).cuda().float()
num_vertices_full = adj_VF.shape[0]
num_triangles = adj_VF.shape[1]
num_evec = opt.num_evec
LBO_gt = torch.from_numpy(x['L']).cuda().double()
LBO_gt = LBO_gt.unsqueeze(0).expand(opt.batchSize, num_vertices_full, num_vertices_full)
triv = torch.from_numpy(x['F'].astype(int)).cuda().unsqueeze(0).expand(opt.batchSize, num_triangles, 3)
triv = triv - 1  # zero based index
vertices = torch.from_numpy(x['V']).cuda().unsqueeze(0).expand(opt.batchSize, num_vertices_full, 3)
phi = torch.from_numpy(x['Phi']).cuda().double() #OH
phi = phi[:,:num_evec].unsqueeze(0).expand(opt.batchSize, num_vertices_full, num_evec)
evals = torch.from_numpy(x['Lambda']).cuda().double()
evals = evals[:num_evec].unsqueeze(0).expand(opt.batchSize, num_evec, 1).transpose(2, 1)
area_V = torch.from_numpy(x['A']).cuda().double().unsqueeze(0).expand(opt.batchSize, num_vertices_full, 1)

# =============Loading======================================== #
x = sio.loadmat(opt.partial_shape_path)
part = torch.from_numpy(x['partial_shape'])
num_vertices_part = part.shape[0]
part = part.cuda().unsqueeze(0).expand(opt.batchSize, num_vertices_part, 6).double()

# =============DEFINE stuff for logs ======================================== #
# Launch visdom for visualization
vis = visdom.Visdom(port=8888, env=opt.env)
save_path = opt.save_path
dir_name = os.path.join('./log/', save_path)
if not os.path.exists(dir_name):
    os.makedirs(dir_name, True)
logname = os.path.join(dir_name, 'log.txt')

opt.manualSeed = 1  # random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
optimization_loss = []


# ===================CREATE optimizer================================= #
lrate = 0.01 # learning rate
pointsReconstructed = torch.randn((2, num_vertices_full, 3), requires_grad=True, device="cuda", dtype = torch.float)
optimizer = optim.Adam([{'params' : pointsReconstructed}], lr=lrate)

# OH: calculate ground truth diffusion distance
gt = vertices.cuda().float()
D_diff_gt = calc_euclidean_dist_matrix(phi)

# =============start of the optimization loop ======================================== #
for epoch in range(opt.nepoch):
    if (epoch % 100) == 99:
        lrate = lrate/2.0
        optimizer = optim.Adam([{'params': pointsReconstructed}], lr=lrate)

    optimizer.zero_grad()

    if epoch == 0:
        pointsReconstructed.data = gt[:,:,:3].data.clone() # initialization of the optimized mesh
        pointsReconstructed.data = pointsReconstructed.data + 0.05*torch.randn_like(pointsReconstructed.data)


    #Calulcate diffusion distance for the optimized mesh
    L_reconstructed, area_matrix, area_matrix_inv, W = LBO(pointsReconstructed.double(), triv.long())
    L_sym = torch.bmm(area_matrix ** 0.5, torch.bmm(L_reconstructed, area_matrix_inv ** 0.5))
    K = torch.tensor(num_evec)
    N = torch.tensor(num_vertices_full)
    phi_reconstructed_sym, lambda_reconstructed = Eigendecomposition(L_sym[0].cpu().double(), K, N)
    phi_reconstructed_sym = phi_reconstructed_sym.unsqueeze(0).expand(opt.batchSize, num_vertices_full, num_evec).cuda()

    phi_reconstructed = torch.bmm(area_matrix_inv ** 0.5, phi_reconstructed_sym) #convert from eigenfunctions of L_sym to L
    square_norm_phi_reconstructed = torch.bmm(phi_reconstructed.transpose(2,1), torch.bmm(area_matrix, phi_reconstructed))
    square_norm_phi_reconstructed = torch.diag(square_norm_phi_reconstructed[0,:,:])
    square_norm_phi_reconstructed = square_norm_phi_reconstructed.unsqueeze(0).unsqueeze(1)
    phi_reconstructed = phi_reconstructed/(square_norm_phi_reconstructed ** 0.5)
    D_diff_rec = calc_euclidean_dist_matrix(phi_reconstructed)
    loss_diff = torch.mean((D_diff_rec - D_diff_gt) ** 2)

    #Calculate chamfer distance from the partial shape to the optimized mesh
    d_chamfer = torch.mean(distChamfer(part[:, :, :3], pointsReconstructed.double()) ** 2)

    loss_net = loss_diff + 100*d_chamfer
    loss_net.backward()
    optimizer.step()  # gradient update


    # VIZUALIZE
    vis.scatter(X=pointsReconstructed[0,:,:3].data.cpu(), win='Optimized mesh',
                opts=dict(title="Optimized mesh",markersize=2, ), )
    vis.scatter(X=gt[0,:,:3].data.cpu(), win='full input mesh',
                opts=dict(title="full input mesh", markersize=2, ), )
    vis.scatter(X=part[0, :, :3].data.cpu(), win='partial input part',
                opts=dict(title="partial input part", markersize=2, ), )
    save_model = pointsReconstructed.cpu().detach().numpy()
    sio.savemat('optimization_result.mat',
                {'pointsReconstructed': save_model})

    print('[%d: %d] train loss:  %f' % (epoch, loss_net,  loss_net.item()))

    # UPDATE CURVES
    optimization_loss.append(loss_net.item())

    vis.line(X=np.arange(len(optimization_loss)),
             Y=np.array(optimization_loss),
             win='loss',
             opts=dict(title="loss", legend=["L2curve_train_smpl"]))

    vis.line(X=np.arange(len(optimization_loss)),
             Y=np.log(np.array(optimization_loss)),
             win='log',
             opts=dict(title="log", legend=["L2curve_train_smpl" + opt.env]))


