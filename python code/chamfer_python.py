import torch


def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    P = rx.t() + ry - 2 * zz
    return P


def NN_loss(x, y, dim=0):
    dist = pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    return values.mean()


def distChamfer(a, b):
    x, y = a, b
    bs_x, num_points_x, points_dim_x = x.size()
    bs_y, num_points_y, points_dim_y = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind_x = torch.arange(0, num_points_x).type(torch.cuda.LongTensor)
    diag_ind_y = torch.arange(0, num_points_y).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(2).expand_as(zz)
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = rx + ry - 2 * zz
    return torch.min(P, 2)[0] #torch.min(P, 1)[0], torch.min(P, 2)[0] , torch.min(P, 1)[1], torch.min(P, 2)[1]

