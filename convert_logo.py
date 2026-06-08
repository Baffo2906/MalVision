from PIL import Image, ImageOps

logo = Image.open("logo.png").convert("RGB")
logo = ImageOps.invert(logo)
logo.save("logo_dark.png")