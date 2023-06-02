import os

image_folder = "F:\keras\kerascv-demo\images"
default_images = {}
    
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        with open(image_path, "rb") as f:
            default_images[filename] = f.read()

print(default_images['building.jpg'])
    
