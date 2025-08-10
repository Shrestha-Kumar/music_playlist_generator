# 🎵 Music Playlist Generator

A **content-based music recommendation system** built with Python that analyzes songs using **MFCC (Mel-Frequency Cepstral Coefficients)** features and recommends similar tracks using **cosine similarity**.

## 📌 Overview

This project processes audio files (from datasets like [FMA - Free Music Archive](https://github.com/mdeff/fma)) to extract audio features, stores them for reuse, and generates playlists of similar songs based on a selected track.

**Key features:**
- Automatically extracts **80 MFCC-based statistical features** from each song (mean, max, min, std).
- Uses **cosine similarity** to find songs that sound most alike.
- Caches processed data to speed up subsequent runs.
- Handles corrupted/unreadable audio files gracefully.
- Works with the FMA small dataset (or similar datasets).

---

## 🛠️ Technologies Used
- **Python 3.x**
- [NumPy](https://numpy.org/) – Numerical computations
- [Librosa](https://librosa.org/) – Audio processing & MFCC extraction
- [scikit-learn](https://scikit-learn.org/) – Cosine similarity
- [joblib](https://joblib.readthedocs.io/) – Saving/loading processed data
- [glob](https://docs.python.org/3/library/glob.html) & [os](https://docs.python.org/3/library/os.html) – File handling

---

## 📂 Project Structure

```
📦 music-playlist-generator
 ┣ 📜 main.py                # Main script (your provided code)
 ┣ 📜 features.pkl           # Cached extracted features (auto-generated)
 ┣ 📜 similarity_vectors.pkl # Cached similarity matrix (auto-generated)
 ┣ 📜 title.pkl              # Cached mapping of indices to song file paths (auto-generated)
 ┣ 📂 fma_small/             # Audio dataset (e.g., FMA small)
 ┗ 📜 README.md              # Project documentation
```

---

## 🚀 How It Works

1. **Feature Extraction**
   - Loads `.mp3` files from the dataset folders (e.g., `./fma_small/000/`).
   - Extracts **20 MFCC coefficients** for each song.
   - Calculates **mean, max, min, std** for each coefficient → **80 features total**.

2. **Similarity Calculation**
   - Computes **cosine similarity** between all feature vectors.
   - Stores results in a similarity matrix.

3. **Recommendation**
   - Given a song index, finds the most similar songs and generates a playlist.

---

## 📥 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/music-playlist-generator.git
   cd music-playlist-generator
   ```

2. **Install dependencies**
   ```bash
   pip install numpy librosa scikit-learn joblib
   ```

3. **Download Dataset**
   - Get the [FMA small dataset](https://github.com/mdeff/fma) (8,000 tracks, 30s clips).
   - Extract into the `./fma_small/` folder.

4. **Run the script**
   ```bash
   python main.py
   ```

---

## 🎯 Usage Example

```python
# Example: Generate recommendations for song at index 7996
playlist_generator(song_index=7996, 
                   feature_vectors=similarity_vectors, 
                   titles=title_dict)
```

**Sample Output:**
```
🎶🎶 Recommended songs based on your selection 🎶🎶

      1. track_000.mp3
      2. track_001.mp3
      3. track_002.mp3
      ...
```

---

## ⚠️ Notes
- The **first run** may take a long time as it processes and extracts features from all audio files.
- Processed features are **cached** in `.pkl` files for faster future runs.
- You can change the number of recommendations by modifying:
  ```python
  num_recommendations=10
  ```

---

## 📜 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements
- [Free Music Archive (FMA)](https://github.com/mdeff/fma) for the dataset.
- [Librosa](https://librosa.org/) for audio feature extraction tools.
