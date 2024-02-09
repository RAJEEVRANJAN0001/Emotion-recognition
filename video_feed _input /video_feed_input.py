import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load your pre-trained model
model = load_model('trained_modal".h5"file/emotion_model_final_2.h5 copy')  

# Define colors for rectangles based on emotions
emotion_colors = {
    'Angry': (0, 0, 255),  # Red
    'Disgust': (0, 255, 0),  # Green
    'Fear': (0, 255, 255),  # Yellow
    'Happy': (255, 0, 0),  # Blue
    'Sad': (255, 0, 255),  # Purple
    'Surprise': (255, 255, 0),  # Orange
    'Neutral': (255, 255, 255),  # White
}

# Function to process each frame and perform emotion recognition
def process_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_color = frame[y:y + h, x:x + w]

        # Preprocess frame for CNN input
        resized_frame = cv2.resize(roi_color, (48, 48))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        normalized_frame = gray_frame / 255.0
        input_data = np.expand_dims(np.expand_dims(normalized_frame, axis=-1), axis=0)

        # Make predictions using the trained CNN model
        prediction = model.predict(input_data)
        emotion_label = np.argmax(prediction)

        # Define emotion labels (you may need to adjust this based on your model's classes)
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        predicted_emotion = emotion_labels[emotion_label]

        # Get the color for the rectangle based on the predicted emotion
        rectangle_color = emotion_colors[predicted_emotion]

        # Display the predicted emotion and draw a colored rectangle
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rectangle_color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)

    return frame

# Input video file path
video_path = "video_feed _input " 

# Open the video file
video = cv2.VideoCapture(video_path)

# Get video properties
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object (let OpenCV choose based on the file extension)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for MP4 with H.264
out = cv2.VideoWriter('emotion_video.mp4', fourcc, fps, (width, height)) 

while True:
    # Capture video frame
    ret, frame = video.read()

    if not ret:
        break  # Break the loop if the video ends

    # Process the frame
    processed_frame = process_frame(frame)

    # Save the frame to the output video
    out.write(processed_frame)

    # Display the video frame with face detection and emotion recognition
    cv2.imshow('Emotion Recognition', processed_frame)

    if cv2.waitKey(1) == 27:  # Press 'Esc' key to exit the loop
        break

# Release resources
video.release()
out.release()
cv2.destroyAllWindows()
