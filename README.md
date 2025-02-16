# Face Recognition and Online Proctoring System

## Overview
This project detects and recognizes faces from a scene and provides the recognized person's name as a voice output. It consists of three main modules:

- **Creator**: Captures and stores face datasets.
- **Trainer**: Trains the model using the collected face data.
- **Recognizer**: Recognizes faces in real-time and provides a voice output of the person's name.

## Modules

### 1. Creator Module
This module is responsible for creating the dataset.
- Takes the person's ID and name as input.
- Detects faces using OpenCV's `haarcascade_frontalface_default.xml`.
- Captures 50-100 grayscale images for better feature extraction.
- Stores face images in a dataset folder.

### 2. Trainer Module
Trains the model using Local Binary Patterns Histograms (LBPH) algorithm.
- Loads face images from the dataset.
- Extracts the ID and associates it with the corresponding images.
- Trains the model using `train()` and saves the trained data in a `.yml` file.
- Handles errors like `IOError: cannot identify image file '*Data\Thumbs.db*'` by either deleting or ignoring the file.

### 3. Recognizer Module
Recognizes faces in real-time and provides a voice output.
- Detects faces from the scene.
- Uses `predict()` to identify the face and returns an ID with confidence level.
- Maps the ID to a name and uses `pyttsx` for voice output.
- Runs continuously until the user quits.

## Tools Used
- **Python 2.7.14**
- **OpenCV 2.4.13**
- **Pyttsx** (for text-to-speech)
- **Numpy**
- **Pillow**
