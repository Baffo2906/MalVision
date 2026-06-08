import torch
import sys
import os
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch import nn
from torchvision import models, transforms
from PIL import Image
from pipeline.binary_to_image import binary_to_image


def classify(input_path, model_path, already_converted):

    classes = {
        0: "Adialer.C",
        1: "Agent.FYI",
        2: "Allaple.A",
        3: "Allaple.L",
        4: "Alueron.gen!J",
        5: "Autorun.K",
        6: "C2LOP.P",
        7: "C2LOP.gen!g",
        8: "Dialplatform.B",
        9: "Dontovo.A",
        10: "Fakerean",
        11: "Instantaccess",
        12: "Lolyda.AA1",
        13: "Lolyda.AA2",
        14: "Lolyda.AA3",
        15: "Lolyda.AT",
        16: "Malex.gen!J",
        17: "Obfuscator.AD",
        18: "Rbot!gen",
        19: "Skintrim.N",
        20: "Swizzor.gen!E",
        21: "Swizzor.gen!I",
        22: "VB.AT",
        23: "Wintrim.BX",
        24: "Yuner.A",
        25: "benign"
    }

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Linear(2048, 26)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()


    image_path = f"/tmp/{uuid.uuid4()}.png"

    if already_converted:
        image_path = input_path
    else:
        binary_to_image(input_path, image_path)

    # =========================
    # preprocessing
    # =========================

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)

    predicted_index = outputs.argmax().item()
    probabilities = torch.softmax(outputs, dim=1)
    confidence = probabilities[0][predicted_index].item()
    
    print(f"Confidence: {confidence:.2%}")

    return classes[predicted_index], image_path, confidence