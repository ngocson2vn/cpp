"""
export PYTHONPATH=/path/to/main_model_dir
export RUN_STAGE=runtime
"""

import importlib
mod = importlib.import_module("main_model")
model = mod.trainer
for name, _ in model.named_parameters():
  print(name)