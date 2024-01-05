import streamlit as st
import os
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.Conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.Conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.Conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(500, 100)  # Calculate input size for fully connected layer
        self.fc2 = nn.Linear(100, 75)  # Additional layer
        self.fc3 = nn.Linear(75, 10)   # New output layer
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.Conv1(x), 2))
        x = F.relu(F.max_pool2d(self.Conv2_drop(self.Conv2(x)), 2))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Pass through additional layer
        x = F.log_softmax(self.fc3(x), dim=1)  # New output layer with log_softmax
        return x
model = CNN()
model.load_state_dict(torch.load("model/model.pth"))
model.eval()

# Function to make predictions
def predict(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path)
    image = transform(image)
    grey = transforms.Grayscale()
    image = grey(image)
    image=image.unsqueeze(0)
    a=model(image)
    return int(torch.argmax(a))


# Streamlit app

st.title("Digit Recognition")
image = Image.open("images/digit.jpeg")
st.image(image,use_column_width=True)
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    # Perform prediction
    class_index = predict(uploaded_file)
    # Display the result
    st.write(f"Prediction: {class_index}")
    # Feedback Radio Button
    correct = st.radio("Is the prediction made correct?", ["Yes", "No"])
    if correct == 'Yes':
        st.write("")
    else:
        st.write("Uh..ohh...")
        with st.form("form"):
            option = st.selectbox("What should have been the actual nummber according to you?", ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))
            st.write(f"selected: {option}")
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.write("Thanks for The Feedback. We will try to improve our model soon.")
                folder_name = f"data/Feedback/{option}"
                os.makedirs(folder_name, exist_ok=True)
                file_path = os.path.join(folder_name, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
