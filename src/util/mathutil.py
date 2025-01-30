# Copied from https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/math.py

import torch
import torch.nn.functional as F


def soft_ce(pred, target, cfg):
    """Computes the cross entropy loss between predictions and soft targets."""
    pred = F.log_softmax(pred, dim = -1)
    target = two_hot(target, cfg)
    return -(target * pred).sum(-1, keepdim = True)


def log_std(x, low, dif):
    return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps, log_std):
    """Compute Gaussian log probability."""
    residual = -0.5 * eps.pow(2) - log_std
    log_prob = residual - 0.9189385175704956
    return log_prob.sum(-1, keepdim = True)


def squash(mu, pi, log_pi):
    """Apply squashing function."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
    log_pi = log_pi - squashed_pi.sum(-1, keepdim = True)
    return mu, pi, log_pi


def int_to_one_hot(x, num_classes):
    """
    Converts an integer tensor to a one-hot tensor.
    Supports batched inputs.
    """
    one_hot = torch.zeros(*x.shape, num_classes, device = x.device)
    one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return one_hot


def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device = x.device, dtype = x.dtype)
    bin_idx = bin_idx.long()
    soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot = soft_two_hot.scatter(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
    return soft_two_hot


def two_hot_inv(x, cfg):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symexp(x)
    dreg_bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device = x.device, dtype = x.dtype)
    x = F.softmax(x, dim = -1)
    x = torch.sum(x * dreg_bins, dim = -1, keepdim = True)
    return symexp(x)


def gumbel_softmax_sample(p, temperature = 1.0, dim = 0):
    logits = p.log()
    # Generate Gumbel noise
    gumbels = (
        -torch.empty_like(logits, memory_format = torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    return y_soft.argmax(-1)

def categorical_kl(p1: torch.Tensor, p2:torch.Tensor) -> torch.Tensor:
    """
    calculates KL between two Categorical distributions. 
    Copied from the Github https://github.com/daisatojp/mpo/blob/master/mpo/mpo.py 
    :param p1: (B, D) the first distribution
    :param p2: (B, D) the second distribution
    """
    #avoid zero division
    p1 = torch.clamp_min(p1, 0.0001)  
    p2 = torch.clamp_min(p2, 0.0001)  
    
    return torch.mean((p1 * torch.log(p1 / p2)).sum(dim=-1))
    
def gaussian_kl(μi: torch.Tensor, μ: torch.Tensor, Ai: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Decoupled KL between two multivariate gaussian distributions f (updated policy) and g (previous policy).

    C_μ = KL(g(x|μi,Σi)||f(x|μ,Σi))
    C_Σ = KL(g(x|μi,Σi)||f(x|μi,Σ))
    :param μi: (B, n) mean fixed to the mean of the previous policy g
    :param μ: (B, n) mean of the updated policy f
    :param Ai: (B, n, n) lower triangular matrix of the covariance of the previous policy g
    :param A: (B, n, n) lower triangular matrix of the covariance of the updated policy f
    :return: C_μ, C_Σ: scalar
        mean and covariance terms of the KL
    :return: mean of determinanats of Σi, Σ
    ref : https://stanford.edu/~jduchi/projects/general_notes.pdf page.13
    """
    def bt(m):
        return m.transpose(dim0=-2, dim1=-1)

    def btr(m):
        return m.diagonal(dim1=-2, dim2=-1).sum(-1)
    
    n = A.size(-1)
    μi = μi.unsqueeze(-1)  # (B, n, 1)
    μ = μ.unsqueeze(-1)  # (B, n, 1)
    Σi = Ai @ bt(Ai)  # (B, n, n)
    Σ = A @ bt(A)  # (B, n, n)
    
    # Trying to fix inf values via logdet
    Σi_logdet = Σi.logdet()  # (B,)
    Σ_logdet = Σ.logdet()  # (B,)
    
    # Inverse of the covariance matrices
    Σi_inv = Σi.inverse()  # (B, n, n)
    Σ_inv = Σ.inverse()  # (B, n, n)
    
    # Inner terms of the KL divergence
    inner_μ = ((μ - μi).transpose(-2, -1) @ Σi_inv @ (μ - μi)).squeeze()  # (B,)
    inner_Σ = Σ_logdet - Σi_logdet - n + btr(Σ_inv @ Σi) # (B,)
    
    # Mean and covariance terms of the KL divergence
    C_μ = 0.5 * torch.mean(inner_μ)
    C_Σ = 0.5 * torch.mean(inner_Σ)
    
    return C_μ, C_Σ
