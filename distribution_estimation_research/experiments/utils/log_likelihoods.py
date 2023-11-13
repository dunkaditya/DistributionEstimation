import torch
import torch.nn as nn 

log_softmax = nn.LogSoftmax(dim=1)

def get_predictions(test_sets, f_set):
    yhat_set = []
    for i in range(len(test_sets)):
        print("Testing " + str(i) + "th iteration")
        curr_xs = test_sets[i].x
        yhat_set.append(f_set[i](curr_xs))
    return yhat_set
    
def get_log_likelihood(test_sets, f_set):
    yhat_set = get_predictions(test_sets, f_set)
    total_log_softmax = 0
    for i in range(len(yhat_set)):
        total_log_softmax += sum(log_softmax(yhat_set[i]).detach())
    return total_log_softmax[0]