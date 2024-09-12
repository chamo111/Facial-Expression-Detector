import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('emotion_detection_model.h5')

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  
        
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA) 

        roi = roi_gray.astype('float') / 255.0  
        roi = img_to_array(roi)  
        roi = np.expand_dims(roi, axis=0)  

        prediction = model.predict(roi)[0]
        emotion = emotion_labels[np.argmax(prediction)]  

        label_position = (x, y-10)
        cv2.putText(frame, emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()  
    if not ret:
        break

    frame = detect_emotion(frame)

    cv2.imshow('Facial Expression Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

