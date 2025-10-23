# stSurvTrans

The code for Identifying prognosis-associated spatial patterns by integrating bulk RNA-seq and spatial transcriptomic data

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/Miaoyx323/stSurvTrans.git
cd stSurvTrans
pip install -r requirements.txt
```

## Quick start

The following is a basic usage example of stSurvTrans, demonstrating how to load data, train the model, and save the results.

```python
from stSurvTrans import stSurvTrans

if __name__ == "__main__":
    st_path = 'path/to/your/spatial/transcriptomics/data/'
    bulk_path = 'path/to/your/adata_bulk.h5ad'

    model = stSurvTrans(
        st_path=st_path, 
        bulk_path=bulk_path
    )

    model.trainVAE()
    model.trainWeibull()

    model.save()
```

### Function description

1. **Data Loading**: By specifying `st_path` (spatial transcriptomics data path) and `bulk_path` (bulk RNA-seq data path), the model will automatically load and preprocess the data.
2. **VAE Training**: The `trainVAE()` method is used to train the variational autoencoder and learn the latent representation of the data.
3. **Weibull model training**: The `trainWeibull()` method trains the Weibull model based on the output of the VAE for survival analysis.
4. **Model saving**: The `save()` method saves the trained model parameters locally for subsequent use.

### Matters need attention

1. Please ensure that the input data format meets the requirements (spatial transcriptomics data should be in a standard directory structure, and survival data is required for bulk data).
2. The training process may take a long time, depending on the amount of data and hardware configuration. It is recommended to use GPU acceleration (make sure that PyTorch has correctly configured GPU support).
3. If you need to adjust model parameters (such as the number of training epochs, learning rate, etc.), you can refer to the detailed documentation or source code of the stSurvTrans class for custom settings.