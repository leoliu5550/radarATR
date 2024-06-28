import pytest,sys,dynamic_yaml
sys.path.append(".")
import src
from core import YAMLConfig,GLOBAL_CONFIG
cfg = YAMLConfig("config/runtime.yml")
print(cfg.yaml_cfg['model'])
print(GLOBAL_CONFIG)
print(cfg.model)

