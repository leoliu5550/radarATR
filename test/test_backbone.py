import pytest,sys,dynamic_yaml
sys.path.append(".")
import src.nn.backbone 
from core import YAMLConfig,GLOBAL_CONFIG,create
cfg = YAMLConfig("config/runtime.yml")

create(cfg.yaml_cfg['model'])

