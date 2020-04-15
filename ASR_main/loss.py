import numpy as np
import torch

def GaussianLogDensity(x, mu, log_var):
    c = torch.log(torch.tensor(2*np.pi))
    var = torch.exp(log_var)
    x_mu2_over_var = ((x - mu) ** 2 ) / (var + 1e-6)
    log_prob = -0.5 * (c + log_var + x_mu2_over_var)
    return torch.mean(log_prob)

def KLD_loss(mu,logvar):
    # Assume target is N(0,1)
    mu2 = torch.zeros_like(mu)
    logvar2 = torch.zeros_like(logvar)
    var = torch.exp(logvar)
    var2 = torch.exp(logvar2)
    mu_diff_sq = (mu - mu2) ** 2

    dimwise_kld = 0.5*( (logvar2 - logvar) + (var + mu_diff_sq)/(var2 + 1e-6) - 1.)
    return torch.mean(dimwise_kld)

def CrossEnt_loss(logits_cat, y_true):
    '''
    logits_cat: softmax output
    y_true: onehot vector
    '''
    loss = (-y_true*torch.log(logits_cat + 1e-6)).sum(dim=1).mean()
    return loss

def clf_CrossEnt_loss(logits, y_true):
    '''
    logits: output without softmax
    y_true: label(indices)
    '''
    cross_entropy = nn.CrossEntropyLoss()
    loss = cross_entropy(logits, y_true)
    return loss

def entropy_loss(logits):
    loss = torch.mean(-logits*logits.log())
    return loss
