import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import torch
from torch import nn
from typing import Optional


def _reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Reduce loss
    Parameters
    ----------
    loss : torch.Tensor, [batch_size, num_classes]
        Batch losses.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    Returns
    -------
    loss : torch.Tensor
        Reduced loss.
    """
    if reduction == 'elementwise_mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f'{reduction} is not a valid reduction')


def cumulative_link_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                         reduction: str = 'elementwise_mean',
                         class_weights: Optional[np.ndarray] = None,
                         *args, **kwargs) -> torch.Tensor:
    """
    Calculates the negative log likelihood using the logistic cumulative link
    function.
    See "On the consistency of ordinal regression methods", Pedregosa et. al.
    for more details. While this paper is not the first to introduce this, it
    is the only one that I could find that was easily readable outside of
    paywalls.
    Parameters
    ----------
    y_pred : torch.Tensor, [batch_size, num_classes]
        Predicted target class probabilities. float dtype.
    y_true : torch.Tensor, [batch_size, 1]
        True target classes. long dtype.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.
    Returns
    -------
    loss: torch.Tensor
    """
    eps = 1e-15
    likelihoods = torch.clamp(torch.gather(y_pred, 1, y_true), eps, 1 - eps)
    neg_log_likelihood = -torch.log(likelihoods)

    if class_weights is not None:
        # Make sure it's on the same device as neg_log_likelihood
        class_weights = torch.as_tensor(class_weights,
                                        dtype=neg_log_likelihood.dtype,
                                        device=neg_log_likelihood.device)
        neg_log_likelihood *= class_weights[y_true]

    loss = _reduction(neg_log_likelihood, reduction)
    return loss


class CumulativeLinkLoss(nn.Module):
    """
    Module form of cumulative_link_loss() loss function
    Parameters
    ----------
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.
    """

    def __init__(self, reduction: str = 'elementwise_mean',
                 class_weights: Optional[torch.Tensor] = None,
                 *args, **kwargs) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        return cumulative_link_loss(y_pred, y_true,
                                    reduction=self.reduction,
                                    class_weights=self.class_weights)


def _neg_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def create_loss(loss_name, **kwargs):
    if loss_name == 'CLUB':
        return CLUB(**kwargs)
    elif loss_name == 'MTL':
        return MTL(**kwargs)
    elif loss_name == 'CE':
        return CrossEnropy(**kwargs)
    elif loss_name == 'FL':
        return FocalLoss(**kwargs)
    elif loss_name == 'FLA':
        return FocalLossAdaptive(**kwargs)
    else:
        raise ValueError(f'Not support loss {loss_name}.')


class CrossEnropy(nn.Module):
    def __init__(self, normalized=False, reduction='mean', **kwargs):
        super(CrossEnropy, self).__init__()
        self.reduction = reduction
        self.eps = 1e-7
        self.normalized = normalized

    def pc_logsoftmax(self, x, stats):
        numer = exp_x = torch.exp(x)
        demon = stats * exp_x
        _ps = numer / demon
        _pls = torch.log(_ps + self.eps)
        return _pls

    def forward(self, input, target, normalized=None, alpha=None, *args, **kwargs):
        normalized = self.normalized if normalized is None else normalized
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1).type(torch.int64)

        if alpha is not None:
            _alpha = alpha
            if not isinstance(_alpha, torch.Tensor):
                _alpha = torch.tensor(_alpha)
        else:
            _alpha = None

        # For binary classification
        if input.dim() == 1 or (input.dim() == 2 and input.shape[1] == 1):
            if not normalized:
                logpt = F.logsigmoid(input)
            else:
                logpt = torch.log(input)
        else:  # Multi-class
            if not normalized:
                logpt = F.log_softmax(input, dim=-1)
            else:
                logpt = torch.log(input)
            logpt = logpt.gather(1, target)

        logpt = logpt.view(-1)

        loss = -logpt

        if _alpha is not None:
            if _alpha.type() != input.data.type():
                _alpha = _alpha.type_as(input.data)
            if len(_alpha.shape) == 1:
                at = _alpha.gather(0, target.data.view(-1))
            elif len(_alpha.shape) == 2:
                at = _alpha.gather(1, target).view(-1)
            else:
                raise ValueError(f'Not support alpha with dim = {len(_alpha.shape)}.')
            # at = _alpha
            loss = loss * Variable(at)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    def __init__(self, normalized=False, gamma=0, reduction='mean', **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-7
        self.normalized = normalized

    def forward(self, input, target, normalized=None, alpha=None, *args, **kwargs):
        normalized = self.normalized if normalized is None else normalized
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1).type(torch.int64)

        if alpha is not None:
            _alpha = alpha
            if not isinstance(_alpha, torch.Tensor):
                _alpha = torch.tensor(_alpha)
        else:
            _alpha = None

        # For binary classification
        if input.dim() == 1 or (input.dim() == 2 and input.shape[1] == 1):
            if not normalized:
                logpt = F.logsigmoid(input)
            else:
                logpt = torch.log(input)
        else:  # Multi-class
            if not normalized:
                logpt = F.log_softmax(input, dim=-1)
            else:
                logpt = torch.log(input)
            logpt = logpt.gather(1, target)

        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * ((1 - pt) ** self.gamma) * logpt

        if _alpha is not None:
            if _alpha.type() != input.data.type():
                _alpha = _alpha.type_as(input.data)
            if len(_alpha.shape) == 1:
                at = _alpha.gather(0, target.data.view(-1))
            elif len(_alpha.shape) == 2:
                at = _alpha.gather(1, target).view(-1)
            else:
                raise ValueError(f'Not support alpha with dim = {len(_alpha.shape)}.')
            # at = _alpha
            loss = loss * Variable(at)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLossAdaptive(nn.Module):
    def __init__(self, normalized=False, gamma=0, reduction='mean', **kwargs):
        super(FocalLossAdaptive, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-7
        self.normalized = normalized
        ps = [0.2, 0.5]
        gammas = [5.0, 3.0]
        i = 0
        self.gamma_dic = {}
        for p in ps:
            self.gamma_dic[p] = gammas[i]
            i += 1

    def get_gamma_list(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            if (pt_sample >= 0.5):
                gamma_list.append(self.gamma)
                continue
            # Choosing the gamma for the sample
            for key in sorted(self.gamma_dic.keys()):
                if pt_sample < key:
                    gamma_list.append(self.gamma_dic[key])
                    break
        return torch.tensor(gamma_list).to(pt.device)

    def forward(self, input, target, normalized=None, alpha=None, *args, **kwargs):
        normalized = self.normalized if normalized is None else normalized
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1).type(torch.int64)

        if alpha is not None:
            _alpha = alpha
            if not isinstance(_alpha, torch.Tensor):
                _alpha = torch.tensor(_alpha)
        else:
            _alpha = None

        # For binary classification
        if input.dim() == 1 or (input.dim() == 2 and input.shape[1] == 1):
            if not normalized:
                logpt = F.logsigmoid(input)
            else:
                logpt = torch.log(input)
        else:  # Multi-class
            if not normalized:
                logpt = F.log_softmax(input, dim=-1)
            else:
                logpt = torch.log(input)
            logpt = logpt.gather(1, target)

        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        gamma = self.get_gamma_list(pt)

        loss = -1 * ((1 - pt) ** gamma) * logpt

        if _alpha is not None:
            if _alpha.type() != input.data.type():
                _alpha = _alpha.type_as(input.data)
            if len(_alpha.shape) == 1:
                at = _alpha.gather(0, target.data.view(-1))
            elif len(_alpha.shape) == 2:
                at = _alpha.gather(1, target).view(-1)
            else:
                raise ValueError(f'Not support alpha with dim = {len(_alpha.shape)}.')
            # at = _alpha
            loss = loss * Variable(at)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MTL(nn.Module):
    def __init__(self, normalized=False, reduction='mean', **kwargs):
        super(MTL, self).__init__()

        self.reduction = reduction
        self.eps = 1e-7
        self.normalized = normalized

    def forward(self, input, target, tau, normalized=None, alpha=None, *args, **kwargs):
        normalized = self.normalized if normalized is None else normalized
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1).type(torch.int64)

        if alpha is not None:
            _alpha = alpha
            if not isinstance(_alpha, torch.Tensor):
                _alpha = torch.tensor(_alpha)
        else:
            _alpha = None

        # For binary classification
        if input.dim() == 1 or (input.dim() == 2 and input.shape[1] == 1):
            if not normalized:
                logpt = F.logsigmoid(input)
            else:
                logpt = torch.log(input)
        else:  # Multi-class
            if not normalized:
                logpt = F.log_softmax(input, dim=-1)
            else:
                logpt = torch.log(input)
            logpt = logpt.gather(1, target)

        logpt = logpt.view(-1)

        log_var = -torch.log(tau)
        loss = - logpt * tau + log_var / 2.0

        if _alpha is not None:
            if _alpha.type() != input.data.type():
                _alpha = _alpha.type_as(input.data)
            if len(_alpha.shape) == 1:
                at = _alpha.gather(0, target.data.view(-1))
            elif len(_alpha.shape) == 2:
                at = _alpha.gather(1, target).view(-1)
            else:
                raise ValueError(f'Not support alpha with dim = {len(_alpha.shape)}.')
            # at = _alpha
            loss = loss * Variable(at)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CLUB(nn.Module):
    def __init__(self, normalized=False, reduction='mean', **kwargs):
        super(CLUB, self).__init__()

        self.reduction = reduction
        self.eps = 1e-7
        self.normalized = normalized

    def forward(self, input, target, tau, normalized=None, alpha=None, *args, **kwargs):
        normalized = self.normalized if normalized is None else normalized

        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1).type(torch.int64)

        if alpha is not None:
            _alpha = alpha
            if not isinstance(_alpha, torch.Tensor):
                _alpha = torch.tensor(_alpha)
        else:
            _alpha = None

        # For binary classification
        if input.dim() == 1 or (input.dim() == 2 and input.shape[1] == 1):
            if not normalized:
                logpt = F.logsigmoid(input)
            else:
                logpt = torch.log(input)
        else:  # Multi-class
            if not normalized:
                logpt = F.log_softmax(input, dim=-1)
            else:
                logpt = torch.log(input)
            logpt = logpt.gather(1, target)

        C = input.shape[-1]
        logpt = logpt.view(-1)
        loss = - logpt * tau + (1 - tau) * np.log(C)

        if _alpha is not None:
            if _alpha.type() != input.data.type():
                _alpha = _alpha.type_as(input.data)
            if len(_alpha.shape) == 1:
                at = _alpha.gather(0, target.data.view(-1))
            elif len(_alpha.shape) == 2:
                at = _alpha.gather(1, target).view(-1)
            else:
                raise ValueError(f'Not support alpha with dim = {len(_alpha.shape)}.')
            # at = _alpha
            loss = loss * Variable(at)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
