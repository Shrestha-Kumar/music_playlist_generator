import numpy as np
import librosa
import os
import glob
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# --- Core Functions ---

def playlist_generator(song_index, feature_vectors, titles, num_recommendations=10):
    """
    Generates and prints a playlist of recommended songs based on a given song.

    Args:
        song_index (int): The index of the song in the dataset to base recommendations on.
        feature_vectors (np.ndarray): The matrix of similarity scores.
        titles (dict): A dictionary mapping indices to song file paths.
        num_recommendations (int): The number of songs to recommend.
    """
    # Get the similarity scores for the given song and sort them in descending order
    indices = np.argsort(feature_vectors[song_index])[::-1]
    
    playlist = []
    # Start from 1 to skip the song itself (which has a similarity of 1.0)
    for i in range(1, num_recommendations + 1):
        playlist.append(os.path.basename(titles[indices[i]]))
    
    # Using set removes any potential duplicate entries, then convert back to a list
    final_playlist = list(set(playlist))
    
    print()
    print("🎶🎶 Recommended songs based on your selection 🎶🎶")
    print()
    for index, song_title in enumerate(final_playlist):
        print(f"      {index + 1}. {song_title}")

# --- Main Script Logic ---

# Define the file to use as a cache to check if processing has already been done
PROCESSED_DATA_FILE = "title.pkl"

if not os.path.exists(PROCESSED_DATA_FILE):
    print("Processed data not found. Starting feature extraction...")
    
    # --- 1. Locate all audio files ---
    base_folder = "./fma_small/"
    all_audio_files = []
    
    # The FMA dataset is organized in numbered folders (e.g., 000, 001, ..., 155)
    # This loop generates the correct folder names with leading zeros.
    for i in range(156):
        # zfill(3) ensures the number is 3 digits long (e.g., 9 -> "009")
        folder_name = str(i).zfill(3)
        search_path = os.path.join(base_folder, folder_name, "*.mp3")
        all_audio_files.extend(glob.glob(search_path))

    # --- 2. Extract features from each audio file ---
    # This is a computationally expensive process.
    
    # Initialize a numpy array to store features. 
    # 80 features = 4 stats (mean, max, min, std) * 20 MFCCs
    feature_array = np.zeros((len(all_audio_files), 80))
    
    successfully_processed_files = {} # To store paths of files that were processed correctly

    for i, file_path in enumerate(all_audio_files):
        try:
            # Load audio file
            y, sr = librosa.load(file_path)
            
            # Extract Mel-Frequency Cepstral Coefficients (MFCCs)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # Calculate feature statistics across the time axis of the MFCCs
            mean_mfcc = np.mean(mfcc, axis=1)
            max_mfcc = np.max(mfcc, axis=1)
            min_mfcc = np.min(mfcc, axis=1)
            std_mfcc = np.std(mfcc, axis=1)
            
            # Combine all stats into a single feature vector for the song
            feature_array[i] = np.concatenate((mean_mfcc, max_mfcc, min_mfcc, std_mfcc))
            
            # If successful, add the file path to our dictionary
            successfully_processed_files[i] = file_path
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1} / {len(all_audio_files)} audio files")

        except Exception as e:
            # If a file is corrupted or cannot be read, print an error and skip it
            print(f"Could not process file: {os.path.basename(file_path)}. Error: {e}")
            continue

    # --- 3. Clean and process the extracted data ---
    
    # Remove any rows that are still all zeros (due to processing errors)
    # This ensures the similarity calculation doesn't fail.
    final_features = feature_array[np.any(feature_array != 0, axis=1)]

    # Calculate the cosine similarity between all songs based on their features
    similarity_vectors = cosine_similarity(final_features)

    # --- 4. Save the processed data to disk for future use ---
    print("\nSaving processed data to disk...")
    joblib.dump(final_features, "features.pkl")
    joblib.dump(similarity_vectors, "similarity_vectors.pkl")
    joblib.dump(successfully_processed_files, "title.pkl")
    print("Data saved successfully.")

else:
    # If the data file exists, load it from the disk instead of re-processing
    print("Found pre-processed data. Loading from disk...")
    title_dict = joblib.load("title.pkl")
    similarity_vectors = joblib.load("similarity_vectors.pkl")
    print("Data loaded successfully.")

# --- Generate a Playlist ---
# Example: Generate recommendations for the song at index 7996
if 'similarity_vectors' in locals() and len(similarity_vectors) > 2:
    playlist_generator(song_index=2, 
                       feature_vectors=similarity_vectors, 
                       titles=title_dict)
else:
    print("\nCould not generate playlist. Not enough data or data not loaded.")