from dataset_loader import get_dataloader
import torch
from torch import nn, optim
from torchvision import models

def train(data_path, epochs, batch_size):
    #carica dataloader
    dataloader = get_dataloader(data_path, batch_size)
    
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    #congelo layer, addestro solo ultimo layer
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    #sostituisco ultimo layer
    model.fc = nn.Linear(2048, 26)
    
    #loss e optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #training loop
    for epoch in range(epochs):
        print(f"[TRAINING] - epoch {epoch} starting")
        for batch in dataloader:
            
            images, labels = batch
            
            #azzero grad
            optimizer.zero_grad()
            
            #forward pass
            outputs = model(images)
            
            #calcolo loss
            loss = criterion(outputs, labels)
            
            #backward pass
            loss.backward()
            
            #aggiorno pesi
            optimizer.step()
            
        print(f"[TRAINING] - epoch {epoch} ended")
        print(f"Loss value: {loss.item()}")
    
    #salva modello
    torch.save(model.state_dict(), "output_model.pth") 
    
    
train("/Users/raffaele/MalVision/malimg_dataset/train", 3, 32)
            
            
    
    
    
    