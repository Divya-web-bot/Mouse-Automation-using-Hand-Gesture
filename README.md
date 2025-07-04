# Mouse-Automation-using-Hand-Gesture
This project enables users to control mouse operations such as cursor movement, click, double-click, and scrolling using real-time hand gestures detected via webcam input. It leverages MediaPipe, OpenCV, and PyAutoGUI to track and interpret hand gestures and map them to mouse actions.
 Cursor Movement based on hand motion
FEATURES: -
👆 Single Finger → Click

✌️ Two Fingers → Double Click

🤟 Three Fingers → Scroll Up

✋ Four Fingers → Scroll Down

🛑 Gesture Locking based on stillness detection to reduce noise

📹 Real-time hand tracking using MediaPipe

 TECHNOLOGIES & LIBRARIES USED: -
 Open CV
 MediaPipe
 PyAytoGUI
 math

 SYSTEM ARCHITECTURE: -
 1. Webcam feed is captured using OpenCV.
 2. Hand Landmarks are detected using MediaPipe
 3. Finger Counting determines gesture type
 4. Gesture Mapping executes mouse actions using PyAutoGUI
 5. Stillness Detection locks/unlocks movement to ensure smooth control
 6. Visual Feedback is shown on-screen for gesture and status

TEAM MEMBERS AND THEIRV WORK: -
1. MAHAK KUMARI: - GESTURE RECOGNITION
2. KOMAL KUMARI: - GESTURE MAPPING TO MOUSE ACTION
3. KHUSHBOO KUMARI: - COORDINATE MAPPING
4. DIVYA KUMARI:- SMOOTH MOTION AND GESTURE FILTERING
5. DIVYANI KUMARI: - USER INTERFACE AND SYSTEM INTEGRATION

