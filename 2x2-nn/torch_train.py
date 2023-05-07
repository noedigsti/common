import torch
from torch_model import CNN, train_model, load_dataset, device


def main():
    images, labels = load_dataset()
    model = CNN().to(device)
    model = train_model(model, images, labels)
    torch.save(model.state_dict(), "./output/torch_trained.pth")


if __name__ == "__main__":
    main()
