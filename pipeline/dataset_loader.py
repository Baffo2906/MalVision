import torch
from torchvision import datasets, transforms

def get_dataloader(data_path, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #valori std normalizzazione
    ])
    
    #prendi dataset
    try:
        dataset = datasets.ImageFolder(data_path, transform)
    except FileNotFoundError:
        raise FileNotFoundError(f"File non trovato: {data_path}")
    
    #dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    
    return dataloader
    