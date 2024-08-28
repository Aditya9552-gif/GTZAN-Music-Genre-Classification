import streamlit as st
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from collections import Counter
from pytube import YouTube
import subprocess

# # Disable TensorFlow logging
# tf.get_logger().setLevel('ERROR')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN custom operations warning

# Load saved models
feature_model1 = load_model('ann_feature_10sec_model.h5')
feature_model2 = load_model('cnn_feature_10sec_model.h5')
spectrogram_model1 = load_model('CNN_spectrogram_10sec_model.h5')
spectrogram_model2 = load_model('GRU_spectrogram_10sec_model.h5')
spectrogram_model3 = load_model('LSTM_spectrogram_10sec_model.h5')

# Load scaler and label encoder for feature-based models
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to extract features from audio
def extract_features_segment(audio_data, sr):
    chroma_stft_mean = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sr))
    chroma_stft_var = np.var(librosa.feature.chroma_stft(y=audio_data, sr=sr))
    rms_mean = np.mean(librosa.feature.rms(y=audio_data))
    rms_var = np.var(librosa.feature.rms(y=audio_data))
    spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
    spectral_centroid_var = np.var(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
    spectral_bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))
    spectral_bandwidth_var = np.var(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))
    rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
    rolloff_var = np.var(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
    zero_crossing_rate_mean = np.mean(librosa.feature.zero_crossing_rate(audio_data))
    zero_crossing_rate_var = np.var(librosa.feature.zero_crossing_rate(audio_data))
    harmony_mean = np.mean(librosa.effects.harmonic(y=audio_data))
    harmony_var = np.var(librosa.effects.harmonic(y=audio_data))
    perceptr_mean = np.mean(librosa.effects.percussive(y=audio_data))
    perceptr_var = np.var(librosa.effects.percussive(y=audio_data))
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_vars = np.var(mfccs, axis=1)

    # Combine the extracted features into a single array
    return [chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, spectral_centroid_mean,
            spectral_centroid_var, spectral_bandwidth_mean, spectral_bandwidth_var, rolloff_mean,
            rolloff_var, zero_crossing_rate_mean, zero_crossing_rate_var, harmony_mean, harmony_var,
            perceptr_mean, perceptr_var, tempo] + list(mfcc_means) + list(mfcc_vars)

# Function to create spectrogram from audio
def create_spectrogram(audio_data, sr, temp_dir, input_shape):
    plt.figure(figsize=(3, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.axis('off')

    # Use the provided temporary directory
    spectrogram_filename = os.path.join(temp_dir, 'spectrogram.png')
    plt.savefig(spectrogram_filename, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    return spectrogram_filename

# def create_spectrogram(audio_data, sr, temp_dir, input_shape):
#     try:
#         plt.figure(figsize=(3, 3))
#         D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
#         librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
#         plt.axis('off')

#         # Use the provided temporary directory
#         spectrogram_filename = os.path.join(temp_dir, 'spectrogram.png')
#         plt.savefig(spectrogram_filename, bbox_inches='tight', pad_inches=0, transparent=True)
#         plt.close()

#         return spectrogram_filename

    # except Exception as e:
    #     print(f"Error creating spectrogram: {e}")
    #     return None


# Function to preprocess input for feature-based models
def preprocess_features(audio_data, sr):
    features = extract_features_segment(audio_data, sr)
    features_scaled = scaler.transform(np.array([features]))
    return features_scaled

# Function to preprocess input for spectrogram-based models
def preprocess_spectrogram(audio_data, sr):
    input_shape = (64, 64, 3)  # Assuming input shape for spectrogram models
    with tempfile.TemporaryDirectory() as temp_dir:
        spectrogram_filename = create_spectrogram(audio_data, sr, temp_dir, input_shape)
        spectrogram = tf.keras.preprocessing.image.load_img(spectrogram_filename, target_size=(input_shape[0], input_shape[1]))
        spectrogram = tf.keras.preprocessing.image.img_to_array(spectrogram)
        spectrogram /= 255.0  # Normalize pixel values to [0, 1]
        spectrogram = np.expand_dims(spectrogram, axis=0)
    return spectrogram

# Function to classify audio segments
def classify_audio_segments(audio_data, sr):
    # Define segment duration (10 seconds)
    segment_duration = 10 * sr

    # Calculate the number of segments
    num_segments = len(audio_data) // segment_duration

    # Define a list to store ensemble predictions
    ensemble_predictions = []

    # Iterate over each segment
    for segment_index in range(num_segments):
        start_sample = segment_index * segment_duration
        end_sample = (segment_index + 1) * segment_duration
        segment_audio = audio_data[start_sample:end_sample]

        # Predictions for each individual model
        features_input = preprocess_features(segment_audio, sr)
        spectrogram_input = preprocess_spectrogram(segment_audio, sr)

        feature_model1_prediction = feature_model1.predict(features_input.reshape(1, features_input.shape[1],))
        feature_model2_prediction = feature_model2.predict(features_input.reshape(1, features_input.shape[1],1))
        spectrogram_model1_prediction = spectrogram_model1.predict(spectrogram_input)
        spectrogram_model2_prediction = spectrogram_model2.predict(spectrogram_input)
        spectrogram_model3_prediction = spectrogram_model3.predict(spectrogram_input)

        # Ensemble prediction using stacking ensemble (meta_model)
        X_meta_test = (feature_model1_prediction + feature_model2_prediction + spectrogram_model1_prediction + spectrogram_model2_prediction + spectrogram_model3_prediction) / 5.0
        # Load the meta-model (stacked ensemble model)
        meta_model = load_model('ensemble_averaging_model.h5')

        # Predict with the meta-model (stacked ensemble)
        ensemble_prediction = meta_model.predict(X_meta_test)
        ensemble_prediction = label_encoder.inverse_transform([np.argmax(ensemble_prediction)])[0]

        # Append the predicted label to the ensemble predictions list
        ensemble_predictions.append(ensemble_prediction)

    # Find the most common prediction
    most_common_ensemble = Counter(ensemble_predictions).most_common(1)[0][0]
    return most_common_ensemble

# Streamlit app
st.title("Music Genre Classification")

# Upload audio file or provide YouTube URL
file_or_url = st.radio("Select method to add audio file:", ('Upload', 'YouTube URL'))

if file_or_url == 'Upload':
    audio_file = st.file_uploader("Upload audio file", type=['mp3', 'wav', 'flac'])
    if audio_file:
        st.audio(audio_file, format='audio/wav')

elif file_or_url == 'YouTube URL':
    youtube_url = st.text_input("Enter YouTube URL:")
    if youtube_url:
        if st.button("Download and Classify"):
            st.write("Downloading audio from YouTube... Please wait")
            yt = YouTube(youtube_url)
            audio_stream = yt.streams.filter(only_audio=True).first()

            # Create a unique temporary directory to store the downloaded file
            temp_dir = tempfile.mkdtemp()
            audio_file_path = audio_stream.download(temp_dir)

            # Convert the downloaded audio to the desired format using ffmpeg command line tool
            output_file_path = os.path.join(temp_dir, 'audio.mp3')
            subprocess.run(['ffmpeg', '-i', audio_file_path, '-ar', '22050', output_file_path])

            # Load the audio file
            audio_data, sr = librosa.load(output_file_path)
            prediction = classify_audio_segments(audio_data, sr)
            st.success("Most Common ensemble_model Prediction: {}".format(prediction))

# Classify button
if st.button("Classify"):
    st.write("Classifying... Please wait")

    # Load the audio file
    audio_data, sr = librosa.load(audio_file)
    prediction = classify_audio_segments(audio_data, sr)
    st.success("Most Common ensemble_model Prediction: {}".format(prediction))
