import torch
import pathlib
import os

SAVE_PATH = pathlib.Path("static")

def read_latest(model:torch.nn.Module):
  os.listdir()

def save(model:torch.nn.Module):
  torch.save(model, SAVE_PATH)
  print('Save done')

def get_device():
  dev = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
  print('Device:', dev)
  return dev


