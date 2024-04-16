# Endovascular Federated Learning

This repository is the implementation of baselines (including UNet, TransUNet and SwinUNet) on the Endovascular Dataset

## Prerequisites

We recommend you to use Python 3.x (1<=x<=9) and conda environment for managing packages: 

```
conda create -n fl-endo python==3.9
conda activate fl-endo
```

Then you can install all the packages with `pip`:
```
conda install pip 
pip install -r requirements.txt
```

## Datasets

We use the Endovascular Dataset obtained from hospitals and laboratories in the UK. The dataset contains X-Ray images come from four modalities, including animal, phantom, simulation and real human. The images are paired with the ground truth annotations. The dataset can be downloaded [here](https://vision.aioz.io/f/7b986782043d403bb50e/). After downloading, please put it into the root folder.


## Parameters 

Here is the detail of parameters used in this repo: 
- `batch-size`: Train dataloader batch size 
- `epochs`: Number of epochs performed at local training 
- `n_parties`: Number of workers (silos) in federated learning setup 
- `comm_round`: Number of communication rounds
- `datadir`: Data directory 
- `logdir` : Log directory
- `modeldir` : Model directory  
- `alg`: Federated learning algorithm: Currently supported: `fedavg`, `fedprox` 
- `lr`: Learning rate. Can be tuned with other values 

## Training 

```
python3 main.py --alg=fedavg --lr=0.01 --mu=5 --epochs=1 --comm_round=50 --n_parties=7 --partition=noniid --beta=0.5 --logdir='./logs/' --datadir='data/phantom_train'
```

## Acknowledgment 

The implementation is based on 
[QinbinLi](https://github.com/QinbinLi/MOON)'s work. We thank the author for sharing the code.

### License

MIT License

