import csv
import os
import random
import sys
from glob import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import random
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

import deep_learning.tianchi.news_classifier.bert.pretrain_args as args
from deep_learning.tianchi.news_classifier.bert.bert_model import BertForMaskedLM, BertConfig

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"