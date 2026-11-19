"""
Prerequisites

pip install safetensors tokenizers
"""

import os
from safetensors.torch import load_file
from tokenizers import Tokenizer

HOME = os.getenv("HOME")
MODEL_DIR = f"{HOME}/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

emb_file = f"{MODEL_DIR}/model-00001-of-00004.safetensors"
print(f"Loading embedding file {emb_file}")
state = load_file(emb_file, device="cuda:0")
embs = state["model.embed_tokens.weight"]

tokenizer = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
encoding = tokenizer.encode("The cat sat on the")
for (token, id) in zip(encoding.tokens, encoding.ids):
  emb = embs[id]
  print(f"{token} -> {id}:\n{emb}\nshape: {emb.shape}")
  print()
