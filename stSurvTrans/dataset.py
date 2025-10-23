from anndata import AnnData

import torch
import torch.utils.data as Data
from torch.nn.functional import one_hot

import random
import numpy as np
import scanpy as sc

from scipy.sparse import csr_matrix
from scipy.optimize import nnls
from scipy.spatial.distance import pdist, squareform

from collections import OrderedDict

def get_neighbor_matrix(
    adata: AnnData, 
    dis_thr: float = 300,
    index_name = '_index'
):
    
    neighbor_matrix = np.zeros((adata.shape[0], adata.shape[0]))

    index_matrix = 0
    index_list = []

    i_adata = adata
    dist_matrix = squareform(pdist(i_adata.obsm['spatial'], metric='euclidean'))
    dist_matrix[dist_matrix <= dis_thr] = 1
    dist_matrix[dist_matrix > dis_thr] = 0
    dist_matrix[np.diag_indices(dist_matrix.shape[0])] = 0
    index_list.extend(list(i_adata.obs[index_name]))
    neighbor_matrix[index_matrix:(i_adata.shape[0] + index_matrix), 
                    index_matrix:(i_adata.shape[0] + index_matrix)] = dist_matrix
    index_matrix += i_adata.shape[0]

    neighbor_matrix = neighbor_matrix[index_list, :]
    neighbor_matrix = neighbor_matrix[:, index_list]
    
    return neighbor_matrix

def set_index(
    adata: AnnData, 
    index_name: str = '_index'
):
    adata.obs[index_name] = list(range(adata.shape[0]))
    return adata

def get_transcriptome_similarity(
    adata: AnnData,
    min_cells=3, 
    index_name='_index', 
    sim_thres=0.8
):
    
    sim_matrix = np.zeros((adata.shape[0], adata.shape[0]))
    
    index_matrix = 0
    index_list = []
    
    tmp = adata.copy()
    # QC
    sc.pp.filter_genes(tmp, min_cells=min_cells)
    sc.pp.normalize_total(tmp)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=2000)
    tmp = tmp[:, tmp.var['highly_variable']]
    sim = np.corrcoef(tmp.X.toarray())
    sim[sim < sim_thres] = 0
    
    index_list.extend(list(tmp.obs[index_name]))
    sim_matrix[index_matrix:(tmp.shape[0] + index_matrix), 
               index_matrix:(tmp.shape[0] + index_matrix)] = sim
    index_matrix += tmp.shape[0]
        
    sim_matrix = sim_matrix[index_list, :]
    sim_matrix = sim_matrix[:, index_list]
    
    return sim_matrix


class vaeDataset(Data.Dataset):
    def __init__(
        self, 
        adata: AnnData, 
        OS_key: str = None, 
        OS_STATUS_key: str = None,
        tech: str = 'sc',
        dis_thr = 300,
        spatial_regulation = True
    ):

        self.anndata = adata
        self.data = csr_matrix(adata.X)
        self.num_data = self.data.shape[0]
        
        # tech process
        self.tech = tech
        
        # index
        self.index_name = 'index_'
        adata = set_index(adata, index_name=self.index_name)
        self.index = adata.obs[self.index_name]
        
        self.pheno_map_dict = {}
        
        self.pheno = [torch.tensor(0.)] * self.num_data
        # self.map_pheno = [torch.tensor(0.)] * self.num_data
        self.OS = [torch.tensor(0.)] * self.num_data
        self.OS_STATUS = [torch.tensor(0.)] * self.num_data
        
        if self.tech == 'bulk':
            self.OS_STATUS = list(adata.obs[OS_STATUS_key].map({'Alive': 0, 'Dead': 1}))
            self.OS = list(adata.obs[OS_key])
            
        if (self.tech == 'spatial') and spatial_regulation:
            self.neighbor_matrix = get_neighbor_matrix(
                adata=adata, 
                dis_thr=dis_thr,
                index_name=self.index_name
            )
            self.sim_matrix = get_transcriptome_similarity(
                adata=adata,
                min_cells=3, 
                index_name=self.index_name
            )
            self.weight_matrix = self.neighbor_matrix * self.sim_matrix
            self.weight_matrix = csr_matrix(self.weight_matrix)
        else:
            # self.weight_matrix = csr_matrix(np.zeros((adata.shape[0], adata.shape[0])))
            self.weight_matrix = None
            self.neighbor_matrix = None
            self.sim_matrix = None
        
    def __getitem__(self, idx):
        return {
            'X': torch.from_numpy(self.data[idx, :].toarray()).unsqueeze(dim=0), 
            'tech': self.tech, 
            'OS': self.OS[idx], 
            'OS_STATUS': self.OS_STATUS[idx],
            'index': self.index[idx]
        }
    
    
    def __len__(self):
        return self.num_data


class weibullDataset(Data.Dataset):
    def __init__(
        self, 
        data, 
        OS,
        OS_STATUS
    ):
        self.data = data
        self.num_data = self.data.shape[0]
        self.OS = OS
        self.OS_STATUS = OS_STATUS
        self.OS_STATUS = self.OS_STATUS.map({'Dead': 1, 'Alive': 0}).values
        
    def __getitem__(self, idx):
        return {
            'X': self.data[idx, :], 
            'OS': self.OS[idx], 
            'OS_STATUS': self.OS_STATUS[idx]
        }
    
    
    def __len__(self):
        return self.num_data


class stDataset(Data.Dataset):
    def __init__(self, data, idx):
        self.data = data
        self.idx = idx
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx, :], self.idx[idx], 0