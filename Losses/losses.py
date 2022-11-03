import torch

def weighted_bce_loss(input, target):
    """
    Loss weighted as given in deeptracking papar

    arguments: predicted and target values
    - log(input) * target - log(1 - input) * (1 - target)

    return : calcualated loss
    <<  Still working on this >>
    """
    eps = 1e-10
    weight = (1 - target)
    
    #loss =   - torch.sum(torch.mul( torch.mul(torch.log( input + eps), target) +   torch.mul( torch.log(1 - input + eps), 1 - target), weight )) / torch.numel(input)
    loss =   - torch.sum( torch.mul(torch.log( input + eps), target) +   torch.mul( torch.log(1 - input + eps), 1 - target) ) / torch.numel(input)

    return loss

def statematrixloss(states, labels):
    eps = 1e-10

    loss =   - torch.sum( torch.mul(torch.log( states + eps), labels) +   torch.mul( torch.log(1 - states + eps), 1 - labels) ) / torch.numel(states)
    #loss =   - torch.sum( torch.mul(torch.mul(torch.log( states + eps), labels) +   torch.mul( torch.log(1 - states + eps), 1 - labels), weight ) ) / torch.numel(states)
    return loss