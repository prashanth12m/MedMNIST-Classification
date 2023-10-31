import streamlit as st
import requests
from PIL import Image
import numpy as np
import keras

# Functions for image classification
bloodmnist_classes = {'0': 'basophil', '1': 'eosinophil', '2': 'erythroblast', '3': 'immature granulocytes(myelocytes, metamyelocytes and promyelocytes)', '4': 'lymphocyte', '5': 'monocyte', '6': 'neutrophil', '7': 'platelet'}

# Define custom metrics
# https://datascience.stackexchange.com/questions/105101/which-keras-metric-for-multiclass-classification
# Include an epsilon term in the denominators to avoid potential divide-by-zero errors: https://www.tensorflow.org/api_docs/python/tf/keras/backend/epsilon

# Recall
def recall_m(y_true, y_pred):
  TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = TP / (Positives+K.epsilon())
  return recall

# Precision
def precision_m(y_true, y_pred):
  TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = TP / (Pred_Positives+K.epsilon())
  return precision

# F1 Score
def f1_m(y_true, y_pred):    
  precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)  
  return 2*((precision*recall)/(precision+recall+K.epsilon()))

def bloodmnist_predict(image_object):
  model = keras.models.load_model('rbmodel-saved-model-24-acc-0.88.hdf5', custom_objects={'precision_m': precision_m, 'recall_m': recall_m, 'f1_m': f1_m})
  # Resize image to 28x28 if needed
  if image_object.size != (28, 28):
    image_object = image_object.resize((28, 28), resample=Image.NEAREST)
  image_array = np.array(image_object)
  image_array = image_array.reshape((1, 28, 28, 3))
  y_pred = model.predict(image_array)
  image_class = np.argmax(y_pred)
  return bloodmnist_classes[str(image_class)]

breastmnist_classes = {'0': 'malignant', '1': 'normal/benign'}
  
def breastmnist_predict(image_object):
  model = keras.models.load_model('cnn_smote-saved-model-16-acc-0.83.hdf5', custom_objects={'precision_m': precision_m, 'recall_m': recall_m, 'f1_m': f1_m})
  image_object_1 = image_object.convert('L') # Convert to greyscale
  # Resize image to 28x28 if needed
  if image_object_1.size != (28, 28):
    image_object_1.resize((28, 28), resample=Image.NEAREST)
  image_array = np.array(image_object_1)
  image_array = image_array/255.0
  image_array = image_array.reshape((1, 28, 28, 1))
  y_pred = model.predict(image_array)
  image_class = np.argmax(y_pred)
  return breastmnist_classes[str(image_class)]
  
# Define the pages in the website

def home():
    st.markdown("# Medical Image Classification Hub")
    st.write('Welcome to the Medical Image Classification Hub.')
    st.write('Please select the category from the menu in the sidebar to continue.')

def blood():
    st.markdown("# Blood Sample Classification")
    st.write('Please upload an image of the blood sample you wish to classify.')
    # Add an image upload widget
    uploaded_image_bl = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'], key='blood')
    # Check if an image has been uploaded
    if uploaded_image_bl is not None:
        with st.spinner('Uploading the image...'):
            # Open the uploaded image using PIL
            image_bl = Image.open(uploaded_image_bl)
            # Display the uploaded image
            st.image(image_bl, caption='Uploaded image', use_column_width=False)
        # Add a 'Predict' button
        predict_button_bl = st.button('Predict Class')
        if predict_button_bl == True:
            # Predict the image class
            with st.spinner('Classifying the image...'):
                predicted_class_bl = bloodmnist_predict(image_bl)
            # Display the predicted class label
            st.success('Predicted Label: {}'.format(predicted_class_bl))

def breast():
    st.markdown("# Breast Scan Classification")
    st.write('Please upload an image of the breast scan you wish to classify.')
    # Add an image upload widget
    uploaded_image_br = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'], key='breast')
    # Check if an image has been uploaded
    if uploaded_image_br is not None:
        with st.spinner('Uploading the image...'):
            # Open the uploaded image using PIL
            image_br = Image.open(uploaded_image_br)
            # Display the uploaded image
            st.image(image_br, caption='Uploaded image', use_column_width=False)
        # Add a 'Predict' button
        predict_button_br = st.button('Predict Class')
        if predict_button_br == True:
		    # Predict the image class
            with st.spinner('Classifying the image...'):
                predicted_class_br = breastmnist_predict(image_br)
            # Display the predicted class label
            st.success('Predicted Label: {}'.format(predicted_class_br))
            

# Define the page options in the sidebar dropdown
page_names_to_funcs = {
    "Home": home,
    "Blood": blood,
    "Breast": breast,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
