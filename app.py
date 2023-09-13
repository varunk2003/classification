import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub
import PIL.Image as Image

# Load your trained ResNet model
model_path = "model.h5"
resnet_model = tf.keras.models.load_model(model_path,custom_objects={"KerasLayer":hub.KerasLayer})

# Function to make predictions
def predict_class(image):
    # Preprocess the image
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    result = resnet_model.predict(image)
    predicted_class_index = np.argmax(result)
    return predicted_class_index

# Streamlit web app
st.title("Image Classification with ResNet")
st.sidebar.title("Upload an Image")

# Upload an image
uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.sidebar.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make predictions when a button is clicked
    if st.sidebar.button("Predict"):
        # Open and preprocess the uploaded image
        pred_image = Image.open(uploaded_image)
        predicted_class_index = predict_class(pred_image)

        # Define class labels
        class_labels = ["GOld Fish", "Harbor Heal", "Jelly Fish", "Lobster", "Oyster", "Sea Turtle", "Squid", "Star Fish"]

        # Display the predicted class label
        st.write("Predicted Class Label:", class_labels[predicted_class_index])

# You can run the Streamlit app with the following command:
# streamlit run your_app_name.py
