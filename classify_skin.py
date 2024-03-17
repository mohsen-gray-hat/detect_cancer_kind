import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import os
import cv2

def predict_skin_cancer(image_path, model_path, class_names):
    # Define transformations for the input image
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the model
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess the input image
    image = Image.open(image_path)
    original_image = cv2.imread(image_path)
    original_height, original_width, _ = original_image.shape
    image = image_transforms(image).unsqueeze(0)

    # Perform prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
    
     # Draw text on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 0, 0)  # Black color
    lineType = 2
    original_image = cv2.resize(original_image, (original_width // 2, original_height // 2))  # Resize the image
    cv2.putText(original_image, predicted_class, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

    # Display the image with the predicted class
    cv2.imshow('Predicted Image', original_image)
    
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return predicted_class

# Example usage
class_names = ['Actinic keratosis', 'Basal cell carcinoma', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Pigmented benign keratosis', 'Seborrheic keratosis', 'Squamous cell carcinoma', 'Vascular lesion']
images=os.listdir(os.getcwd()+'\\'+'images')
for image in images:
    predicted_class = predict_skin_cancer(f'images\\{image}', "skin_cancer_resnet18.pth", class_names)
    print("Predicted skin cancer type:", predicted_class)
