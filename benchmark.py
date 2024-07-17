import torch
import os
from FNN import ReluNet
import training


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def main():
    training.train_model()


if __name__ == "__main__":
    main()
