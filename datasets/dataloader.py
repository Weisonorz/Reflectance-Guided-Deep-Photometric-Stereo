import os, sys
from random import shuffle

import torch 
import torch.nn as nn 
import numpy as np 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import yaml 





def get_loader(dataset: Dataset, batch_size: int, val_split: float, random_seed: int) -> tuple[Dataset, Dataset, DataLoader, DataLoader]:
	"""
	Args:
		dataset: Base dataset
		batch_size: batchsize
		val_split: The percentage of data for validation
		random_seed: The random_seed for splitting data
		
	Returns:
		train_dataset: The training dataset after splitting 
		val_dataset: The validation dataset after splitting 
		train_loader: The dataloader for train_dataset
		val_laoder: The dataloader for val_dataset
	""" 
	generator = torch.Generator().manual_seed(random_seed) 
	train_dataset, val_dataset = random_split(dataset,[1.0-val_split, val_split]) 
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size =batch_size, shuffle=False) 
	return train_dataset, val_dataset, train_loader, val_loader






