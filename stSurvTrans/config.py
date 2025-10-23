# config.py - configuration settings for stSurvTrans package
import torch
import random
from tqdm.notebook import tqdm

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# Global configuration settings
def set_global_config(verbosity=3, random_seed=42, plt_style="seaborn-v0_8"):
    """
    Set global environment configurations (scanpy logs, random seed, plotting style)
    
    Parameters:
        verbosity (int): scanpy log verbosity level (0=no logs, 3=detailed logs)
        random_seed (int): Random seed (to ensure experiment reproducibility)
        plt_style (str): matplotlib plotting style
    """
    # Set scanpy logs
    sc.settings.verbosity = verbosity
    # Set random seed (numpy + torch + random)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    # Set plotting style
    plt.style.use(plt_style)
    sns.set_context("paper")  # seaborn context configuration (paper/ talk/ poster)