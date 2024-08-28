# Music Genre Classification using GTZAN Dataset

This repository contains the code and resources for a music genre classification project. The project utilizes the GTZAN music genre dataset to build models that classify music tracks into different genres. The project includes feature-based models, spectrogram-based models, and an ensemble model that combines predictions from multiple models. The project also includes a Streamlit app for user interaction.

The project is deployed and can be accessed online at [GTZAN Music Genre Classification](https://gtzan-music-genre-classification-lsaugfxgu5urtab8syukyy.streamlit.app/).

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Ensemble Model](#ensemble-model)
- [Deployment](#deployment)
- [File Structure](#file-structure)

## Project Overview
This project focuses on classifying music tracks into genres using the GTZAN music genre dataset. The dataset contains 10 genres with 100 tracks each, all 30 seconds long. The project involves extensive preprocessing, feature extraction, and the development of various machine learning models, including Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) like LSTM and GRU. Finally, an ensemble model is created to improve prediction accuracy.

## Features
- **Audio Feature Extraction:** Extracts various audio features such as zero crossing rate, harmonic and perceptual features, tempo, spectral centroid, spectral rolloff, and MFCCs.
- **Spectrogram Image Generation:** Converts audio segments into spectrogram images for image-based classification models.
- **Multiple Models:** Develops ANN, CNN, CNN-GRU, and CNN-LSTM models based on audio features and spectrogram images.
- **Ensemble Model:** Combines predictions from different models to improve classification accuracy.
- **Streamlit App:** A user-friendly interface that allows users to upload an audio file and get a genre prediction.

## Installation

To run this project locally, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/Aditya9552-gif/GTZAN-Music-Genre-Classification.git
   cd gtzan-music-genre-classification

2. Create a virtual environment:
    python -m venv env

3. Activate the virtual environment:
     .\env\Scripts\activate

4. Install the required packages:
     pip install -r requirements.txt

5. Navigate to the final_app.py file and run it to start the local server:
     python final_app.py

     
## Usage
Audio Feature Extraction and Augmentation
*  Use the Music_Genre_Preprocessing_and_Basic_EDA.ipynb notebook to explore the dataset, extract audio features, and create augmented versions of the audio data.

Model Training
*  Use the features_model.ipynb notebook to create and train ANN and CNN models based on extracted audio features.
*  Use the spectrogram_models.ipynb notebook to create and train CNN, CNN-GRU, and CNN-LSTM models based on spectrogram images.

Ensemble Model
*  Use the ensemble_model.ipynb notebook to combine predictions from the trained models. The ensemble model makes predictions for 10-second segments of the audio and outputs the most common genre prediction.

Streamlit App
*  The music_app.py script provides a user interface to upload audio files, classify the genre, and display the results.

## Preprocessing
The preprocessing steps include:

*  Audio Segmentation: Dividing 30-second audio files into 10-second segments.
*  Augmentation: Creating three augmented versions of each audio segment.
*  Feature Extraction: Extracting audio features like zero crossing rate, harmonic and perceptual features, tempo, spectral centroid, spectral rolloff, and MFCCs.
*  Spectrogram Creation: Generating spectrogram images from the segmented and augmented audio.

## Modeling
Feature-Based Models
*  ANN and CNN Models: Trained on the extracted audio features (mean and variance of various features).
* Notebooks: features_model.ipynb
  
Spectrogram-Based Models
*  CNN, CNN-GRU, and CNN-LSTM Models: Trained on spectrogram images generated from the audio segments.
*  Notebooks: spectrogram_models.ipynb
  
Ensemble Model
*  The ensemble model combines predictions from the above models. It divides an audio file into 10-second segments, makes predictions for each segment, and selects the most common prediction as the final genre prediction.
*  Notebook: ensemble_model.ipynb

## Deployment
The project has been deployed using Streamlit. You can access the live application at GTZAN Music Genre Classification.

To deploy the project yourself, follow these steps:

1. Create an account on Streamlit.
2. Create a new app and link it to your GitHub repository.
3. Set up the environment variables as needed.
4. Deploy the service and access the provided URL.

## File Structure
The repository is structured as follows:

├── Music_Genre_Preprocessing_and_Basic_EDA.ipynb:   Preprocessing and basic EDA notebook

├── features_model.ipynb:                            Feature-based models (ANN, CNN)

├── spectrogram_models.ipynb:                        Spectrogram-based models (CNN, CNN-GRU, CNN-LSTM)

├── ensemble_model.ipynb:                            Ensemble model combining predictions from all models

├── music_app.py:                                    Streamlit app for genre classification

├── label_encoder.pkl:                               Label encoder for both feature and spectrogram models

├── scaler.pkl:                                      Scaler for feature-based models

├── requirements.txt:                                Dependencies

└── README.md:                                       Project readme file
