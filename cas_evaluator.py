# Copyright (c) 2020 Ganler
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from utility.imcliploader import CASEvaluator
from models.experimental import ImagePolicyNet
import os
import torch

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

if __name__ == "__main__":
    model = ImagePolicyNet(n_opt=2).cuda()
    resdir = os.path.join(project_dir, 'trained')
    res = os.path.join(project_dir, 'trained', os.listdir(resdir)[0])
    print(res)
    model.load_state_dict(torch.load(os.path.join(res, 'mlmodel.pth')))
    model.eval()
    
    evaluator = CASEvaluator(folder=os.path.join(project_dir, 'val_data_non_general'))
    pred, skip = evaluator.evaluate(model)
    print(pred, skip)

# Result on Sept.: 
# [0.9777927282651225, 0.9916058020972788, 0.9893865900905597, 0.9942362164982135] 
# [0.49, 0.2311111111111111, 0.29555555555555557, 0.2777777777777778]