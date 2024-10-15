import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification
from torchvision.ops import nms


# Load the model (adjust the path if needed)
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=10,
    ignore_mismatched_sizes=True
)

# Load the trained state_dict (update the path here)
model.load_state_dict(torch.load('/content/drive/MyDrive/sign_language_model.pth'), strict=False)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Streamlit user interface
st.title("Sign Language Recognition")

uploaded_image = st.file_uploader("Upload an image of a sign", type=['png', 'jpg', 'jpeg'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Apply the transformations and prepare the image
    image_tensor = transform(image).unsqueeze(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image_tensor = image_tensor.to(device)

    # Get predictions
    with torch.no_grad():
        output = model(image_tensor).logits
        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.item()

    st.write(f"Predicted class: {predicted_class}")
