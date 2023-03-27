

import torch
import torch.nn as nn
import torch.nn.functional as F

def osp_loss( args, model, model_logits, targets, mu=1. ):
    classes = list(range(0, model.num_classes))
    eps=1e-7
    tol=1e-8

    lmbda = F.relu( model.lambdas )
    epsilon = F.relu( model.epsilons )

    logits = model_logits['logits']
    aux_logits = model_logits['aux_logits']
    aux_loss = F.cross_entropy( aux_logits, targets )
 
    y_input = F.one_hot( targets, model.num_classes )
    y_out   = F.softmax( logits, dim=1 )

    n_pos = torch.sum( y_input, dim=0 ) + 0.1
    n_neg = torch.sum( 1-y_input, dim=0 ) + 0.1

    loss_pos = (1./n_pos) * torch.sum( -y_input * torch.log( y_out + tol ), dim=0 )
    loss_neg = (1./n_neg) * torch.sum( -(1-y_input) * torch.log( 1-y_out + tol ), dim=0 )

    binary_xent = torch.sum(loss_pos + loss_neg) - torch.sum(lmbda*epsilon) + mu * torch.sum(epsilon) 

    #xent = args.alpha * binary_xent + (1-args.alpha) * 0.125 * aux_loss
    xent = args.alpha * binary_xent + (1-args.alpha) * aux_loss
    return xent


class OSPModel(nn.Module):
    def __init__(self, model, num_classes=100, mu=1.):
        super(OSPModel, self).__init__()
        self.num_classes = num_classes

        self.model = model
        self.aux_clf = nn.Linear(model.get_embedding_dim(), num_classes)
        self.epsilons = nn.Parameter( torch.ones(num_classes) * 0.5 )
        self.lambdas = nn.Parameter( torch.ones(num_classes) * (mu - 0.5) )

    def get_message(self):
        return self.model.get_message()

    def get_minimization_vars(self):
        return list( self.model.parameters() ) + list(self.aux_clf.parameters()) + [self.epsilons]

    def get_maximization_vars(self):
        return [ self.lambdas ]

    def forward(self, x):
        features, logits, ft = self.model(x)
        aux_logits = self.aux_clf( features )
        return logits, aux_logits


