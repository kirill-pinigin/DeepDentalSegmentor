import torch

def configure_criterion(criterion_querry : str):
    system_collection = {   "CrossEntropyCriterion" : CrossEntropyCriterion()}
    return system_collection[criterion_querry]


class CrossEntropyCriterion(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def forward(self, result, target):
        return torch.nn.functional.nll_loss(torch.log(result + 1e-9), torch.argmax(target, dim = 1))
