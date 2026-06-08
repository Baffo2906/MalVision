import torch
from torch import nn
from torchvision import models, transforms, datasets
dataset = datasets.ImageFolder("/Users/raffaele/MalVision/malimg_dataset/val")
print(dataset.classes)
from dataset_loader import get_dataloader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate(data_test_path, model_path):
    
    #carico modello con layer addestrato
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.fc = nn.Linear(2048, 26)
    
    #carico pesi salvati
    model.load_state_dict(torch.load(model_path))
    
    model.eval()
    
    dataloader = get_dataloader(data_test_path, batch_size=32)
    
    all_predictions = []
    all_labels = []
    
    for batch in dataloader:
        
        images, labels = batch
        
        #forward pass
        outputs = model(images)
        
        predicted_index = outputs.argmax(dim=1).tolist()
        all_predictions.extend(predicted_index)
        all_labels.extend(labels.tolist())
        
    y_pred = all_predictions
    y_true = all_labels
    
    result = accuracy_score(y_true, y_pred)
    #matrix = confusion_matrix(y_true, y_pred)
    print(f"{classification_report(y_true, y_pred)}")
    
    print(f"{result}")
    #print(f"{matrix}")
    
evaluate("/Users/raffaele/MalVision/malimg_dataset/test", "/Users/raffaele/MalVision/output_model.pth")
        
        
            
    
    

