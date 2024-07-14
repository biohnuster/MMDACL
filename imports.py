import random
import logging
from typing import Optional, Union, List, Dict, Tuple, Any, Callable, Iterable, Literal, Iterator
import itertools
import functools
import math
from datetime import datetime, date, timedelta
import time
from collections import defaultdict, namedtuple, deque, Counter
from pprint import pprint
import pickle
import os
import sys
import dataclasses
from dataclasses import dataclass, asdict
import argparse
import json
import copy
import csv
import re
# import yaml
import threading
import sqlite3
# [END]

# [BEGIN] 第三方常用库
import requests
from tqdm import tqdm
import numpy as np
from numpy import ndarray
import pandas as pd

# import wandb

IntArray = FloatArray = BoolArray = ndarray
# [END]

# [BEGIN] PyTorch相关
import torch
import torch as th
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch import Tensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

import scipy
import scipy.io as sio
import scipy.sparse as sp
import matplotlib.pyplot as plt

# import xgboost as xgb
# import wandb

IntTensor = FloatTensor = BoolTensor = FloatScalarTensor = SparseTensor = Tensor
IntArrayTensor = FloatArrayTensor = BoolArrayTensor = Union[Tensor, ndarray]
# [END]
