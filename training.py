import BatchedDataHandler
import FNN
import torch
import time

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def save_model(model, path):
    torch.save(model, path)


def train_model():

    train_loader_mandelbrot, test_loader_mandelbrot, val_loader_mandelbrot = BatchedDataHandler.get_data("./Mandelbrot.csv",
                                                                                  ["X", "Y"],
                                                                                  ["divergend"],
                                                                                  4096)
    model_0 = FNN.ReluNet().to(device)

    optimizer = torch.optim.Adam(model_0.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    print("Training started!")
    start = time.time()
    FNN.train_model(train_loader_mandelbrot, test_loader_mandelbrot, model_0,optimizer, loss_fn, epochs=1)
    end = time.time()

    print("Benchmark finished in {} seconds".format(end - start))
    # delete referenced for garbage collection for systems with less ram nobody knows how the python gc works XD
    del model_0
    del train_loader_mandelbrot, test_loader_mandelbrot
    torch.cuda.empty_cache()
