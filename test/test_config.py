import pytest,sys,dynamic_yaml
sys.path.append(".")
# importlib.import_module("core")
import core

class TestConfig:
    cfg = core.YAMLConfig("config/runtime.yml")
    def test_Basic(self):
        assert self.cfg.yaml_cfg["train_dataloader"]["type"] == "DataLoader"
        assert self.cfg.yaml_cfg["val_dataloader"]["type"] == "DataLoader"
        assert self.cfg.yaml_cfg["ema"]["warmups"] == 2000
        assert self.cfg.yaml_cfg["experiment"] == "Develop"


