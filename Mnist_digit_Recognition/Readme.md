# Digit Classifier with CNNs on MNIST Dataset: Deployment and Feedback Loop

## Assignment Overview

In this assignment, you will be tasked with training a Digit Classifier using Convolutional Neural Networks (CNNs) on the MNIST dataset. Additionally, you will deploy the model using Streamlit and implement a user feedback loop into the application. The goal is to store wrong predictions in a database or folder for future fine-tuning.

### Prerequisites

1. **Python and Libraries**: Ensure you have Python installed along with the necessary libraries, including TensorFlow/Pytorch, Streamlit, and any other dependencies.

2. **MNIST Dataset**: Familiarize yourself with the MNIST dataset, a collection of 28x28 grayscale images of handwritten digits (0-9). Dataset is already provided in `input` folder

### Task 1: Model Training

1.1. **Data Preprocessing**: Load and preprocess the MNIST dataset. Normalize pixel values and reshape the images.

1.2. **Build CNN Model**: Create a Convolutional Neural Network for digit classification using a framework like TensorFlow/Keras.

1.3. **Compile and Train Model**: Compile the model with an appropriate optimizer and loss function. Train the model on the MNIST dataset.

1.4. **Evaluate Model**: Evaluate the model's performance on a separate test set to ensure accuracy.

### Task 2: Streamlit Deployment

2.1. **Install Streamlit**: Install Streamlit using `pip install streamlit` if you haven't already.

2.2. **Create Streamlit App**: Develop a Streamlit application that allows users to upload images and see the model's predictions.

2.3. **Integrate Model**: Incorporate the trained CNN model into the Streamlit app for real-time predictions.

### Task 3: User Feedback Loop

3.1. **Feedback Interface**: Enhance the Streamlit app to include a feedback mechanism where users can indicate if the model prediction was correct or not.

3.2. **Store Incorrect Predictions**: Implement a functionality to store incorrect predictions in a database or a folder. Save the input image, predicted label, and actual label for each incorrect prediction.

### Task 4: Model Fine-tuning

4.1. **Retrieve Incorrect Predictions**: Develop a script to retrieve the stored incorrect predictions from the database or folder.

4.2. **Re-train Model**: Use the retrieved incorrect predictions to fine-tune the model. Update the model weights based on the feedback.

4.3. **Evaluate Fine-tuned Model**: Evaluate the performance of the fine-tuned model on the test set.

## Submission

You are expected to submit a **zipped folder** with the following files.

1. app.py (streamlit app)
2. model files (torch/tensorflow models)
3. requirements.txt (packages needed to run model)
4. Readme.md (Instructions to run code)
