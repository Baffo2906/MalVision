import torch
from torch import nn
from torchvision import models
from dataset_loader import get_dataloader
from sklearn.metrics import accuracy_score, classification_report

def evaluate(data_test_path, model_path):
    
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.fc = nn.Linear(2048, 26)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    dataloader = get_dataloader(data_test_path, batch_size=32)
    
    all_predictions = []
    all_labels = []
    
    for batch in dataloader:
        images, labels = batch
        outputs = model(images)
        all_predictions.extend(outputs.argmax(dim=1).tolist())
        all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_predictions))
    print(f"Accuracy: {accuracy_score(all_labels, all_predictions):.4f}")

evaluate("malimg_dataset/test", "output_model.pth")
        
        
            
    
    

