import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score
def ROC_curve_micro(pred_tensor, label_tensor):
    n_class=pred_tensor.size(1)
    one_hot_labels = F.one_hot(label_tensor, num_classes=n_class) 
    is_positive = one_hot_labels
    is_negative =1-one_hot_labels
    fn_diff = -is_positive.flatten()
    fp_diff = is_negative.flatten()
    thresh_tensor = -pred_tensor.flatten()
    fn_denom = is_positive.sum()
    fp_denom = is_negative.sum()
    sorted_indices = torch.argsort(thresh_tensor)
    sorted_fp_cum = fp_diff[sorted_indices].cumsum(0) / fp_denom
    sorted_fn_cum = -fn_diff[sorted_indices].flip(0).cumsum(0).flip(0) / fn_denom

    sorted_thresh = thresh_tensor[sorted_indices]
    sorted_is_diff = sorted_thresh.diff() != 0
    sorted_fp_end = torch.cat([sorted_is_diff, torch.tensor([True])])
    sorted_fn_end = torch.cat([torch.tensor([True]), sorted_is_diff])

    uniq_thresh = sorted_thresh[sorted_fp_end]
    uniq_fp_after = sorted_fp_cum[sorted_fp_end]
    uniq_fn_before = sorted_fn_cum[sorted_fn_end]

    FPR = torch.cat([torch.tensor([0.0]), uniq_fp_after])
    FNR = torch.cat([uniq_fn_before, torch.tensor([0.0])])

    return {
        "FPR": FPR,
        "FNR": FNR,
        "TPR": 1 - FNR,
        "min(FPR,FNR)": torch.minimum(FPR, FNR),
        "min_constant": torch.cat([torch.tensor([-1]), uniq_thresh]),
        "max_constant": torch.cat([uniq_thresh, torch.tensor([0])])
    }
def ROC_AUC_micro(pred_tensor, label_tensor):
    roc = ROC_curve_micro(pred_tensor, label_tensor)
    FPR_diff = roc["FPR"][1:]-roc["FPR"][:-1]   
    TPR_sum = roc["TPR"][1:]+roc["TPR"][:-1]
    return torch.sum(FPR_diff*TPR_sum/2.0)
#AUM 
def Proposed_AUM_micro(pred_tensor, label_tensor):

    roc = ROC_curve_micro(pred_tensor, label_tensor)
    min_FPR_FNR = roc["min(FPR,FNR)"][1:-1]
    constant_diff = roc["min_constant"][1:].diff()
    return torch.sum(min_FPR_FNR * constant_diff)
def ROC_curve_macro(pred_tensor, label_tensor):
    n_class=pred_tensor.size(1)
    one_hot_labels = F.one_hot(label_tensor, num_classes=n_class)
    is_positive = one_hot_labels
    is_negative =1-one_hot_labels
    fn_diff = -is_positive
    fp_diff = is_negative
    thresh_tensor = -pred_tensor
    fn_denom = is_positive.sum(dim=0).clamp(min=1)
    fp_denom = is_negative.sum(dim=0).clamp(min=1)
    sorted_indices = torch.argsort(thresh_tensor,dim=0)
    sorted_fp_cum = torch.div(torch.gather(fp_diff, dim=0, index=sorted_indices).cumsum(0), fp_denom)
    sorted_fn_cum = -torch.div(torch.gather(fn_diff, dim=0, index=sorted_indices).flip(0).cumsum(0).flip(0) , fn_denom)
    sorted_thresh = torch.gather(thresh_tensor, dim=0, index=sorted_indices)
    #Problem starts here 
    zeros_vec=torch.zeros(1,n_class)
    FPR = torch.cat([zeros_vec, sorted_fp_cum])
    FNR = torch.cat([sorted_fn_cum, zeros_vec])
    return {
        "FPR_all_classes": FPR,
        "FNR_all_classes": FNR,
        "TPR_all_classes": 1 - FNR,
        "min(FPR,FNR)": torch.minimum(FPR, FNR),
        "min_constant": torch.cat([-torch.ones(1,n_class), sorted_thresh]),
        "max_constant": torch.cat([sorted_thresh, zeros_vec])
    }

def ROC_AUC_macro(pred_tensor, label_tensor):
    roc = ROC_curve_macro(pred_tensor, label_tensor)
    FPR_diff = roc["FPR_all_classes"][1:,:]-roc["FPR_all_classes"][:-1,]
    TPR_sum = roc["TPR_all_classes"][1:,:]+roc["TPR_all_classes"][:-1,:]
    sum_FPR_TPR= torch.sum(FPR_diff*TPR_sum/2.0,dim=0)
    count_non_defined=(sum_FPR_TPR == 0).sum()
    if count_non_defined==pred_tensor.size(1):
        return 0
    return  sum_FPR_TPR.sum()/(pred_tensor.size(1)-count_non_defined)
def Proposed_AUM_macro(pred_tensor, label_tensor):

    roc = ROC_curve_macro(pred_tensor, label_tensor)
    min_FPR_FNR = roc["min(FPR,FNR)"][1:-1,:]
    constant_diff = roc["min_constant"][1:,:].diff(dim=0)
    sum_min= torch.sum(min_FPR_FNR * constant_diff,dim=0)
    count_non_defined=(sum_min== 0).sum()
    if count_non_defined==pred_tensor.size(1):
        return 0
    return  sum_min.sum()/(pred_tensor.size(1)-count_non_defined)

four_labels = torch.tensor([2,2,1,1])

four_pred = torch.tensor([[0.4, 0.3, 0.3],
                         [ 0.2, 0.1, 0.7],
                         [0.5,0.2,0.3],
                         [0.3,0.4,0.3]])