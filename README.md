# Complete Deep Learning with PyTorch

Hands-on notebooks working through deep learning with PyTorch, from tensors and autograd to CNNs, transfer learning, RNNs and LSTMs.

## Structure

```
.
├── data/                              # Local datasets
│   ├── 100_Unique_QA_Dataset.csv      # Small QA dataset (RNN)
│   ├── Signs_Data_Training.h5         # Hand-sign images (train)
│   └── Signs_Data_Testing.h5          # Hand-sign images (test)
├── notebooks/
│   ├── 01_pytorch_fundamentals/       # Tensors, autograd, perceptron, nn.Module, Dataset/DataLoader
│   ├── 02_ann/                        # Feed-forward nets: breast cancer, Fashion-MNIST, Optuna tuning
│   ├── 03_cnn_transfer_learning/      # Hand-sign CNN and VGG-16 transfer learning
│   └── 04_rnn_lstm/                   # RNN question answering, LSTM
├── requirements.txt
└── README.md
```

## Notebooks

| Section | Notebook | Topic |
|---|---|---|
| 01 | `tensor.ipynb` | PyTorch tensor basics |
| 01 | `autograd.ipynb` | Automatic differentiation |
| 01 | `single_percepton.ipynb` | Single perceptron from scratch |
| 01 | `NN_module.ipynb` | Building models with `nn.Module` |
| 01 | `dataset_dataloader.ipynb` | `Dataset` and `DataLoader` |
| 02 | `breast_cancer_self.ipynb` | Neural net built manually |
| 02 | `breast_cancer_nn.ipynb` | Same task using `torch.nn` |
| 02 | `DL_D1.ipynb` | Image classification with torchvision |
| 02 | `FMNIST.ipynb` | Fashion-MNIST ANN + Optuna hyperparameter tuning |
| 02 | `optuna.ipynb` | Hyperparameter tuning experiments |
| 03 | `handsign.ipynb` | CNN on the hand-sign dataset |
| 03 | `handsign_transfer_learning_vgg-16.ipynb` | Transfer learning with VGG-16 |
| 04 | `rnn.ipynb` | RNN-based question answering |
| 04 | `LSTM.ipynb` | LSTM |

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
jupyter lab
```

Run notebooks from their own folder; local data is loaded from `../../data/`.
`FMNIST.ipynb` downloads Fashion-MNIST automatically via torchvision into `data/FashionMNIST/` (not tracked by git).

## License

The code in this repository is released under the [MIT License](LICENSE).
Datasets in `data/` and those downloaded by the notebooks belong to their original authors and keep their own terms.
