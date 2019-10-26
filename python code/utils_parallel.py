import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
import time
import scipy
import scipy.io as sio
from scipy.sparse import csr_matrix, csc_matrix
import os

def calculate_grad_input_matrix_parallel(input_matrix, grad_output, row_ind, col_ind, col_ind_mod_N, col_ind_div_N, eigvals, eigvecs, N, K):
    '''
    All input variables are torch Tensors
    :param input_matrix: [N x N]
    :param input_matrix_support_flatten: vectorization of the rows of the binary support matrix.
                                         Binary support matrix is defined for input_matrix
                                         such that there is 1 in the non-zero entries, and 0 otherwise.
    :param eigvals: [N]
    :param eigvecs: [N x K]
    :param K: number of eigenfunctions
    :param N: number of vertices
    :return:
    grad_input_matrix: [N x N] . The gradient with respect to the input matrix.
    '''

    #approximating pseudo inverse of (lambda_k*Id - input_matrix) for each k, and storing batched pseudo inverse over k
    grad_input_matrix = csr_matrix((1, N ** 2), dtype=np.double)
    inv_lambda_mat = eigvals.unsqueeze(1) - eigvals.unsqueeze(0)
    inv_lambda_mat = 1/inv_lambda_mat
    ind = np.diag_indices(K)
    inv_lambda_mat[ind[0], ind[1]] = torch.zeros(K).double()
    inv_lambda_batch_k = torch.diag_embed(inv_lambda_mat)
    eigvecs_batch = eigvecs.expand(K, N, K)
    pinv_approx_batch = torch.bmm(eigvecs_batch, torch.bmm(inv_lambda_batch_k, eigvecs_batch.transpose(2,1)))

    # TODO: parallelize
    for k in range(K):
        print(k)
        P = pinv_approx_batch[k]

        t = time.time()
        P = csr_matrix(P.cpu().data.numpy())
        elapsed = time.time() - t
        print(elapsed)

        uk = eigvecs[:, k].data.numpy()  # dims: n

        t = time.time()
        data = uk[col_ind_div_N] * np.squeeze(np.asarray(P[row_ind, col_ind_mod_N]))
        elapsed = time.time() - t
        print(elapsed)
        t = time.time()
        Rk = csr_matrix((data, (row_ind, col_ind)), shape=(N, N ** 2))
        elapsed = time.time() - t
        print(elapsed)

        t = time.time()
        grad_input_matrix = grad_input_matrix + csr_matrix(grad_output[:, k], shape=(1, N)).dot(Rk)
        elapsed = time.time() - t
        print(elapsed)

    return grad_input_matrix


USE_PYTORCH_SYMEIG = True
class EigendecompositionParallel(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input_matrix, K, N):
        if USE_PYTORCH_SYMEIG:
            eigvals, eigvecs = torch.symeig(input_matrix, eigenvectors=True)
        else:
            input_matrix_np = input_matrix.data.cpu().numpy()
            Knp = K.data.numpy()
            eigvals, eigvecs = eigh(input_matrix_np, lower=False)        
#             eigvals, eigvecs = eigh(input_matrix_np, eigvals=(0, Knp - 1), lower=False)
            eigvals = torch.from_numpy(eigvals).cuda()
            eigvecs = torch.from_numpy(eigvecs).cuda()
        
        ctx.save_for_backward(input_matrix, eigvals, eigvecs, K, N)
        return (eigvecs[:,:K], eigvals[:K])

    @staticmethod
    def backward(ctx, grad_output, grad_output2):
        # grad_output stands for the grad of eigvecs
        #grad_output2 stands for the grad of eigvecs
        
#         t = time.time()
        input_matrix, eigvals, eigvecs, K, N = ctx.saved_tensors
        
        Kknp = eigvals.shape[0]
        grad_K = None
        grad_N = None
        
        #NOTE: if we suffer memory issues we can split the K eigenvectors and iterate using an accumulator
        Ink = torch.eye(Kknp,K).double().cuda()
        eig_inv = 1/(eigvals[None,:K]-eigvals[:,None] + Ink) * (1-Ink)
        uk = eigvecs[:, :K]
        grad_input_matrix = torch.mm(torch.mm(eigvecs,eig_inv[:,:]*torch.mm(eigvecs.t(),grad_output[:, :K])),uk.t()).t()        
#         print('time -',time.time() - t)
        
        return grad_input_matrix, grad_K, grad_N
    
    
# class EigendecompositionParallel(torch.autograd.Function):

#     # Note that both forward and backward are @staticmethods
#     @staticmethod
#     def forward(ctx, input_matrix, K, N):
#         input_matrix_np = input_matrix.data.numpy()
#         Knp = K.data.numpy()
#         eigvals, eigvecs = eigh(input_matrix_np, eigvals=(0, Knp - 1), lower=False)

#         eigvals = torch.from_numpy(eigvals)
#         eigvecs = torch.from_numpy(eigvecs)
#         ctx.save_for_backward(input_matrix, eigvals, eigvecs, K, N)
#         return (eigvecs, eigvals)

#     # This function has only a single output, so it gets only one gradient
#     @staticmethod
#     def backward(ctx, grad_output, grad_output2):
#         input_matrix, eigvals, eigvecs, K, N = ctx.saved_tensors
#         Knp = K.data.numpy().item()
#         Nnp = N.data.numpy().item()
#         grad_K = None
#         grad_N = None

#         # constructing the indices for the calculation of sparse du/dL
#         #TODO: refactor this call and the same call in optimizzation script
#         x = sio.loadmat("./../data/eigendecomposition/downsampled_tr_reg_004.mat")
#         adj_VV = x['adj_VV']
#         L_mask_flatten = csc_matrix.reshape(adj_VV, (1, Nnp ** 2))
#         _, col_ind = L_mask_flatten.nonzero()
#         Lnnz = col_ind.shape[0]
#         row_ind = np.arange(Nnp)
#         row_ind = np.repeat(row_ind, Lnnz)
#         col_ind = np.tile(col_ind, Nnp)
#         row_ind = row_ind.astype((np.int64))
#         col_ind = col_ind.astype((np.int64))
#         col_ind_mod_N = col_ind % Nnp
#         col_ind_div_N = col_ind // Nnp

#         # TODO: parallelize
#         grad_input_matrix = calculate_grad_input_matrix_parallel(input_matrix, grad_output, row_ind, col_ind, col_ind_mod_N, col_ind_div_N, eigvals, eigvecs, Nnp, Knp)
#         grad_input_matrix = torch.from_numpy(np.transpose(np.reshape(grad_input_matrix.todense(), (Nnp, Nnp))))

#         return grad_input_matrix, grad_K, grad_N


# Aliasing
EigendecompositionParallel = EigendecompositionParallel.apply






