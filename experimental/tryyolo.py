import os, sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from models.backbone import SamplerBackbone

rate_option = [1, 2, 4, 8, 12, 16, 32, 64, 128]
