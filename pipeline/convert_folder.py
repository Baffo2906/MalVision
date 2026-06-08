import os
from binary_to_image import binary_to_image

folder = "/Users/raffaele/MalVision/malimg_dataset/test/benign"

for filename in os.listdir(folder):
    input_path = os.path.join(folder, filename)
    output_path = os.path.join(folder, f"{filename}.png")
    binary_to_image(input_path, output_path)
    os.remove(input_path)

