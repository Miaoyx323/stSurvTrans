import os
import torch
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from itertools import cycle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix

from .model import (
    comVI, Weibull, 
    MMD_loss, loss_func, continuous_weibull_loglik, spatial_regulation, weibull_qf
)
from .config import set_global_config
from .data_preprocessing import preprocess_data, load_single_cell_data
from .dataloader import csDataLoader
from .dataset import (
    vaeDataset, weibullDataset, stDataset,
    get_neighbor_matrix, get_transcriptome_similarity
)

class stSurvTrans:
    def __init__(
            self,
            st_path,
            bulk_path,
            OS_key='OS',
            OS_STATUS_key='OS_STATUS',
            n_latent=64,
            encoder_dims=[1024,256],
            decoder_dims=[256,1024],
            batch_size=128,
            lr=1e-4,
            device='cuda',
            workdir='./workdir',
    ):
        set_global_config()

        self.n_latent = n_latent
        self.batch_size = batch_size
        self.lr = lr
        self.workdir = workdir
        self.device = device

        self.OS_key = OS_key
        self.OS_STATUS_key = OS_STATUS_key

        os.makedirs(self.workdir, exist_ok=True)

        self.adata_st = load_single_cell_data(st_path)
        self.adata_bulk = load_single_cell_data(bulk_path)

        self._check_bulk_data()

        self.adata_st, self.adata_bulk = preprocess_data(
            adata_st=self.adata_st,
            adata_bulk=self.adata_bulk
        )
        
        self.vae = comVI(
            n_in=self.adata_st.n_vars,
            n_latent=n_latent,
            encoder_dims=encoder_dims,
            decoder_dims=decoder_dims
        )

        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=30, 
            gamma=1.0
        )

        self.weibull = Weibull(n_in=n_latent)
        self.optimizer_weibull = torch.optim.Adam(
            self.weibull.parameters(), 
            lr=1e-3,
            weight_decay=1e-1
        )
    
    def _check_bulk_data(self):
        if self.OS_key is None or self.OS_STATUS_key is None:
            raise AssertionError(
                    "Please provide the OS_key and OS_STATUS_key of bulk data."
                )
        else:
            pass
        # Check if OS and OS_STATUS are in adata.obs
        assert self.OS_key in self.adata_bulk.obs.columns, f"OS_key '{self.OS_key}' not found in adata.obs"
        assert self.OS_STATUS_key in self.adata_bulk.obs.columns, f"OS_STATUS_key '{self.OS_STATUS_key}' not found in adata.obs"
        # Check if OS_STATUS values are only 'Alive' and 'Dead'
        assert set(self.adata_bulk.obs[self.OS_STATUS_key].unique()).issubset({'Alive', 'Dead'}), f"OS_STATUS_key '{self.OS_STATUS_key}' must contain only 'Alive' and 'Dead'"
        # Keep only data where OS is greater than 0
        self.adata_bulk = self.adata_bulk[self.adata_bulk.obs[self.OS_key] > 0, :].copy()
    
    def trainVAE(
            self,
            n_epochs_vae=800,
            mmd_weight=100.0,
            gaussian_weight=1.0
    ):
        try:
            dataset_st = vaeDataset(self.adata_st, tech='spatial')
            dataset_bulk = vaeDataset(
                self.adata_bulk, tech='bulk',
                OS_key=self.OS_key, OS_STATUS_key=self.OS_STATUS_key
            )

            dataloader_st = csDataLoader(
                dataset_st,
                batch_size=self.batch_size,
                shuffle=True
            )
            dataloader_bulk = csDataLoader(
                dataset_bulk,
                batch_size=self.batch_size,
                shuffle=True
            )
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")
        
        mmd_func = MMD_loss()
        total_loss_list = []

        self.vae.to(self.device)
        
        pbar = tqdm(range(n_epochs_vae), desc="VAE training progress")
        for epoch in pbar:
            self.vae.train()
            total_loss = 0

            for batch_st, batch_bulk in zip(dataloader_st, 
                                            cycle(dataloader_bulk)):
                
                data_st = batch_st['X']
                data_bulk = batch_bulk['X']
                
                min_batch_size = np.min([data_bulk.size(0), data_st.size(0)])
                
                data_st = batch_st['X'][:min_batch_size]
                tech_label_st = batch_st['tech'][:min_batch_size]
                
                data_bulk = batch_bulk['X'][:min_batch_size]
                tech_label_bulk = batch_bulk['tech'][:min_batch_size]
                
                data_st = data_st.to(self.device)
                data_st = data_st.view(data_st.size(0), -1).to(torch.float32)
                
                data_bulk = data_bulk.to(self.device)
                data_bulk = data_bulk.view(data_bulk.size(0), -1).to(torch.float32)
                
                out_net_st = self.vae(data_st, tech_label_st)
                loss_dict_st = loss_func(
                    output=out_net_st, 
                    x=data_st, 
                    tech_label=tech_label_st,
                    gaussian_weight=gaussian_weight
                )
                
                out_net_bulk = self.vae(data_bulk, tech_label_bulk)
                loss_dict_bulk = loss_func(
                    output=out_net_bulk, 
                    x=data_bulk, 
                    tech_label=tech_label_bulk,
                    gaussian_weight=gaussian_weight
                )
                
                loss_mmd = mmd_func(source=out_net_st['latent'], target=out_net_bulk['latent'])
                
                loss = loss_dict_bulk['total_loss'] + loss_dict_st['total_loss'] + loss_mmd * mmd_weight
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss_dict_bulk['total_loss'].item() + loss_dict_st['total_loss'].item()
                
            self.scheduler.step()
            
            avg_loss = total_loss / self.adata_st.n_obs / self.adata_st.n_vars
            total_loss_list.append(avg_loss)

            pbar.set_description(f"VAE training progress (Epoch {epoch+1}/{n_epochs_vae}, Loss: {avg_loss:.4f})")
        
        plt.figure(figsize=(6,4))
        plt.plot(range(1, n_epochs_vae + 1), total_loss_list, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VAE Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(self.workdir, 'vae_training_loss.png')
        plt.savefig(loss_plot_path, bbox_inches='tight')
        plt.close()
        print(f"VAE training completed, loss curve saved to: {loss_plot_path}")

        self._get_latent()
        self.save_vae_model()

    
    def _get_latent_representation(
            self,
            tech='spatial'
    ):
        self.vae.eval()
        latent_representations = []

        if tech == 'spatial':
            dataloader = csDataLoader(
                vaeDataset(self.adata_st, tech='spatial'),
                batch_size=self.batch_size,
                shuffle=False
            )
        elif tech == 'bulk':
            dataloader = csDataLoader(
                vaeDataset(
                    self.adata_bulk, tech='bulk',
                    OS_key=self.OS_key, OS_STATUS_key=self.OS_STATUS_key
                ),
                batch_size=self.batch_size,
                shuffle=False
            )
        else:
            NotImplementedError("Data type only supports 'spatial' or 'bulk'.")

        with torch.no_grad():
            for batch in dataloader:
                data = batch['X']
                tech_label = batch['tech']
                
                data = data.to(self.device)
                data = data.view(data.size(0), -1).to(torch.float32)
                
                out_net = self.vae(data, tech_label)
                latent_representations.append(out_net['z_mu'].cpu().numpy())
        
        latent_representations = np.vstack(latent_representations)
        return latent_representations
    
    def _get_latent(
            self, 
            save_path=None
    ):
        self.adata_st.obsm['_VAE'] = self._get_latent_representation(tech='spatial')
        self.adata_bulk.obsm['_VAE'] = self._get_latent_representation(tech='bulk')

        if save_path is None:
            save_path = self.workdir

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sc.write(os.path.join(save_path, 'adata_st_with_latent.h5ad'), self.adata_st)
        sc.write(os.path.join(save_path, 'adata_bulk_with_latent.h5ad'), self.adata_bulk)
        print(f"AnnData with computed embeddings saved under {save_path}.")
        
        # adata = self.adata_st.concatenate(self.adata_bulk, batch_categories=['spatial', 'bulk'])
        adata = sc.concat(
            adatas=[self.adata_st, self.adata_bulk],
            label='batch',
            keys=['spatial', 'bulk']
        )

        sc.pp.neighbors(adata, use_rep='_VAE')
        sc.tl.umap(adata)

        plt.rcParams["figure.figsize"] = (4,4)
        umap_save_path = os.path.join(save_path, "umap_plot.png")

        plt.figure(figsize=(6,4))
        sc.pl.umap(adata, color=["batch"], wspace=0.4, 
                   show=False)
        plt.savefig(umap_save_path, bbox_inches='tight')
        plt.close()
        print(f"UMAP plot saved to: {umap_save_path}")
    
    def trainWeibull(
            self, 
            n_epochs_bulk=100,
            n_epochs_weibull=300,
            spatial_weight=1e-2
    ):
        dataset_bulk = weibullDataset(
            data=self.adata_bulk.obsm['_VAE'],
            OS=list(self.adata_bulk.obs['OS'].values),
            OS_STATUS=self.adata_bulk.obs['OS_STATUS']
        )
        dataloader_bulk = DataLoader(
            dataset=dataset_bulk, batch_size=2, shuffle=True
        )

        total_loss_list = []
        self.weibull.to(self.device)

        pbar = tqdm(range(n_epochs_bulk), desc="Weibull training progress (bulk pretraining)")
        for epoch in pbar:
            
            self.weibull.train()
            total_loss = 0

            for batch in dataloader_bulk:
                
                data = batch['X']
                OS = batch['OS']
                OS_STATUS = batch['OS_STATUS']
                
                data = data.to(self.device)
                data = data.view(data.size(0), -1).to(torch.float32)
                
                OS = OS.to(self.device).to(torch.float32)
                OS_STATUS = OS_STATUS.to(self.device).to(torch.int32)
                
                out_OS = self.weibull(data)
                loss = continuous_weibull_loglik(
                    x=OS, c=OS_STATUS, 
                    alpha=out_OS['alpha'],beta=out_OS['beta'], 
                    clip_prob=None
                )
                
                self.optimizer_weibull.zero_grad()
                loss.backward()
                self.optimizer_weibull.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / self.adata_bulk.n_obs
            total_loss_list.append(avg_loss)
            pbar.set_description(f"Weibull training progress (bulk pretraining) (Epoch {epoch+1}/{n_epochs_bulk}, Loss: {avg_loss:.4f})")
        
        dataset_st = stDataset(
            data=self.adata_st.obsm['_VAE'],
            idx=self.adata_st.obs['index_']
        )
        dataloader_st = DataLoader(
            dataset=dataset_st, batch_size=256, shuffle=True
        )

        neighbor_matrix = get_neighbor_matrix(
            adata=self.adata_st, 
            dis_thr=300,
            index_name='index_'
        )
        sim_matrix = get_transcriptome_similarity(
            adata=self.adata_st,
            min_cells=3, 
            index_name='index_'
        )
        weight_matrix = csr_matrix(neighbor_matrix * sim_matrix)

        OS_loss_list = []
        spatial_loss_list = []
        total_loss_list = []
        self.weibull.to(self.device)

        pbar = tqdm(range(n_epochs_weibull), desc="Weibull training progress (joint training)")
        for epoch in pbar:

            self.weibull.train()

            OS_loss = 0
            spatial_loss = 0
            total_loss = 0

            for data_st, idx_st, _ in dataloader_st:
                data_st = data_st.to(self.device)
                data_st = data_st.view(data_st.size(0), -1).to(torch.float32)
                
                out_OS = self.weibull(data_st)
                
                loss = spatial_regulation(
                            features=weibull_qf(0.5, out_OS['alpha'], out_OS['beta']).unsqueeze(1), 
                            idx=idx_st,
                            weight_matrix=weight_matrix,
                            device=self.device,
                            tech=['spatial', 'spatial']
                        ) * spatial_weight
            
                self.optimizer_weibull.zero_grad()
                loss.backward()
                self.optimizer_weibull.step()
                
                spatial_loss += loss.item()
                total_loss += loss.item()
            
            for batch in dataloader_bulk:
                
                data = batch['X']
                OS = batch['OS']
                OS_STATUS = batch['OS_STATUS']
                
                data = data.to(self.device)
                data = data.view(data.size(0), -1).to(torch.float32)
                
                OS = OS.to(self.device).to(torch.float32)
                OS_STATUS = OS_STATUS.to(self.device).to(torch.int32)
                
                out_OS = self.weibull(data)
                
                loss = continuous_weibull_loglik(
                    x=OS, c=OS_STATUS, 
                    alpha=out_OS['alpha'],beta=out_OS['beta'], 
                    clip_prob=None
                )
                
                self.optimizer_weibull.zero_grad()
                loss.backward()
                self.optimizer_weibull.step()
                
                OS_loss += loss.item()
                total_loss += loss.item()
            
            OS_loss_list.append(OS_loss / self.adata_bulk.n_obs)
            spatial_loss_list.append(spatial_loss / self.adata_st.n_obs)
            total_loss_list.append(total_loss / (self.adata_bulk.n_obs + self.adata_st.n_obs))

            pbar.set_description(f"Weibull training progress (joint training) (Epoch {epoch+1}/{n_epochs_weibull}, Loss: {total_loss / (self.adata_bulk.n_obs + self.adata_st.n_obs):.4f})")
        
        plt.figure(figsize=(6,4))
        plt.plot(range(1, n_epochs_weibull + 1), total_loss_list, label='Total Loss')
        plt.plot(range(1, n_epochs_weibull + 1), OS_loss_list, label='OS Loss')
        plt.plot(range(1, n_epochs_weibull + 1), spatial_loss_list, label='Spatial Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Weibull Joint Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        joint_loss_plot_path = os.path.join(self.workdir, 'weibull_joint_training_loss.png')
        plt.savefig(joint_loss_plot_path, bbox_inches='tight')
        plt.close()
        print(f"Weibull joint training completed, loss curve saved to: {joint_loss_plot_path}")

        self.predict_OS()
        self.save_weibull_model()
    

    def _get_OS(
            self, 
            dataloader
    ):
        self.weibull.eval()
        OS_pred = []
        alpha_pred = []
        beta_pred = []

        with torch.no_grad():
            for batch in dataloader:
                data = batch['X']
                
                data = data.to(self.device)
                data = data.view(data.size(0), -1).to(torch.float32)
                
                out_OS = self.weibull(data)

                alpha_batch = out_OS['alpha'].cpu().detach().numpy()
                beta_batch = out_OS['beta'].cpu().detach().numpy()
                OS_batch = weibull_qf(0.5, out_OS['alpha'], out_OS['beta']).cpu().detach().numpy()
                
                OS_pred.extend(OS_batch)
                alpha_pred.extend(alpha_batch)
                beta_pred.extend(beta_batch)
        
        return OS_pred, alpha_pred, beta_pred
        

    def predict_OS(
            self, 
            save_path=None
    ):
        dataset_bulk = weibullDataset(
            data=self.adata_bulk.obsm['_VAE'],
            OS=list(self.adata_bulk.obs['OS'].values),
            OS_STATUS=self.adata_bulk.obs['OS_STATUS']
        )
        dataloader_bulk = DataLoader(
            dataset=dataset_bulk, batch_size=self.batch_size, shuffle=False
        )
        dataset_st = weibullDataset(
            data=self.adata_st.obsm['_VAE'],
            OS=[0.]*self.adata_st.n_obs,
            OS_STATUS=pd.Series(['Dead']*self.adata_st.n_obs)
        )
        dataloader_st = DataLoader(
            dataset=dataset_st, batch_size=self.batch_size, shuffle=False
        )

        self.adata_st.obs['pred_OS'], self.adata_st.obs['pred_alpha'], self.adata_st.obs['pred_beta'] = self._get_OS(dataloader_st)
        self.adata_bulk.obs['pred_OS'], self.adata_bulk.obs['pred_alpha'], self.adata_bulk.obs['pred_beta'] = self._get_OS(dataloader_bulk)

        self.adata_st.obs['risk_score'] = 1 / (1 + self.adata_st.obs['pred_OS'])

        if save_path is None:
            save_path = self.workdir

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sc.write(os.path.join(save_path, 'adata_st_with_predicted_OS.h5ad'), self.adata_st)
        sc.write(os.path.join(save_path, 'adata_bulk_with_predicted_OS.h5ad'), self.adata_bulk)
        print(f"AnnData with predicted survival time saved under {save_path}.")
        
        if 'spatial' in self.adata_st.uns_keys() and 'spatial' in self.adata_st.obsm_keys():
            plt.rcParams["figure.figsize"] = (6,5)
            spatial_save_path = os.path.join(save_path, "spatial_pred_OS_plot.png")
            
            plt.figure(figsize=(10,4))
            sc.pl.spatial(
                self.adata_st, 
                color=["pred_OS", "risk_score"], 
                size=1.5, 
                wspace=0.4, 
                show=False,
            )
            plt.savefig(spatial_save_path, bbox_inches='tight')
            plt.close()
            print(f"Spatial distribution plot saved to: {spatial_save_path}")
    

    def save_model(
            self,
            save_path=None
    ):
        if save_path is None:
            save_path = self.workdir

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        torch.save(self.vae.state_dict(), os.path.join(save_path, 'vae_model.pth'))
        torch.save(self.weibull.state_dict(), os.path.join(save_path, 'weibull_model.pth'))
        print(f"Model parameters saved under {save_path}.")
    

    def save_vae_model(
            self,
            save_path=None
    ):
        if save_path is None:
            save_path = self.workdir

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        torch.save(self.vae.state_dict(), os.path.join(save_path, 'vae_model.pth'))
        print(f"VAE model parameters saved under {save_path}.")
    

    def save_weibull_model(
            self,
            save_path=None
    ):
        if save_path is None:
            save_path = self.workdir

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        torch.save(self.weibull.state_dict(), os.path.join(save_path, 'weibull_model.pth'))
        print(f"Weibull model parameters saved under {save_path}.")


    def save_adata(
            self,
            save_path=None
    ):
        if save_path is None:
            save_path = self.workdir

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        sc.write(os.path.join(save_path, 'adata_st_final.h5ad'), self.adata_st)
        sc.write(os.path.join(save_path, 'adata_bulk_final.h5ad'), self.adata_bulk)
        print(f"Final AnnData saved under {save_path}.")
    

    def save(
            self,
            save_path=None
    ):
        self.save_model(save_path=save_path)
        self.save_adata(save_path=save_path)