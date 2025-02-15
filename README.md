# Emotion Recognition from Video

## Project Overview
This project is an emotion recognition system that detects faces in a video stream and classifies their emotions using a Convolutional Neural Network (CNN). It processes video input, detects faces using OpenCV's Haar Cascade classifier, and predicts emotions using a trained deep learning model.

## Features
- Detects faces in video streams using OpenCV's Haar Cascade.
- Classifies emotions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
- Overlays emotion labels and colored rectangles around detected faces.
- Processes live webcam input or video files.
- Saves the processed video with emotion recognition results.

## Technologies Used
- Python
- OpenCV
- TensorFlow/Keras
- NumPy

## Prerequisites
Before running the project, install the required dependencies:
```sh
pip install opencv-python numpy tensorflow keras
```

## Model Training (Optional)
If you need to train your own emotion recognition model, follow these steps:
1. Gather a dataset of labeled facial expressions (e.g., FER2013).
2. Preprocess images by resizing them to 48x48 and converting them to grayscale.
3. Train a CNN model using TensorFlow/Keras.
4. Save the trained model as `emotion_model.h5`.

## Running the Project
### 1. Using Webcam for Real-time Emotion Detection
Run the following script to start real-time emotion detection:
```sh
python webcam_emotion_recognition.py
```
This script captures frames from the webcam, detects faces, predicts emotions, and displays the results.

### 2. Processing a Video File
To analyze a video file and save the output, run:
```sh
python video_emotion_recognition.py --input video.mp4 --output output.mp4
```
Replace `video.mp4` with your input video file and specify the output file name.

## Explanation of the Code
### 1. Loading Dependencies
- `cv2`: Used for video processing and face detection.
- `numpy`: For numerical operations.
- `tensorflow.keras.models.load_model`: Loads the pre-trained emotion recognition model.

### 2. Face Detection
Haar Cascade classifier detects faces in each frame:
```python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
```

### 3. Emotion Classification
Each detected face is processed and passed to the trained CNN model:
```python
resized_frame = cv2.resize(roi_color, (48, 48))
gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
normalized_frame = gray_frame / 255.0
input_data = np.expand_dims(np.expand_dims(normalized_frame, axis=-1), axis=0)
prediction = model.predict(input_data)
emotion_label = np.argmax(prediction)
```

### 4. Displaying Results
The predicted emotion is displayed on the frame with a color-coded rectangle:
```python
cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rectangle_color, 2)
cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)
```

## Output Example
- **Input:** Video feed with human faces.
- **Output:** The video stream displays emotion labels above detected faces.

## Troubleshooting
- Ensure the correct path for `haarcascade_frontalface_default.xml`.
- Check that the trained model file (`emotion_model.h5`) is correctly loaded.
- Adjust video frame dimensions if needed.

## Future Improvements
- Improve accuracy with a deeper neural network.
- Optimize real-time performance.
- Integrate with voice-based emotion recognition.

## License
This project is open-source and available for educational and research purposes.

## Author
Developed by Rajeev Ranjan Pratap Singh

---
Let me know if you need any modifications! 🚀
