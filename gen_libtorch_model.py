# -*-coding:utf-8-*-
#!/usr/bin/env python3
import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

import torch
import config as cfg
import models.PointNetVlad as PNV

if len(sys.argv) < 3:
    print('Usage: python gen_libtorch_model.py <model_path> <output_model_path>')
    exit(1)
script, resume_filename, output_model_path = sys.argv
INPUT_DIM = 3
INPUT_NUM_POINTS = cfg.NUM_POINTS

checkpoint = torch.load(resume_filename)
saved_state_dict = checkpoint['state_dict']
print(resume_filename)

amodel = PNV.PointNetVlad(global_feat=True, feature_transform=False, max_pool=False,
                          output_dim=cfg.FEATURE_OUTPUT_DIM, num_points=INPUT_NUM_POINTS)

amodel.load_state_dict(saved_state_dict)

amodel.eval()
example = torch.rand(1, 1, INPUT_NUM_POINTS, INPUT_DIM)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(amodel, example)
traced_script_module.save(output_model_path)
