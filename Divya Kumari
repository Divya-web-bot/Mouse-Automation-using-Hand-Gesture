import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Movement settings
MOVEMENT_SCALE = 2.25
SMOOTHING_FACTOR = 0.7
GESTURE_HOLD_THRESHOLD = 10  # Frames to maintain gesture lock
MIN_MOVEMENT = 0.002  # Threshold to filter noise
STILLNESS_THRESHOLD = 0.008  # Threshold to lock for minimal movement
STILLNESS_FRAMES = 15  # Frames of stillness required to lock

# State variables
movement_locked = False
lock_position = None
gesture_hold_counter = 0
last_position = None
stillness_counter = 0
last_movement = 0

def count_raised_fingers(landmarks):
    raised = 0
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]
    
    for tip, pip in zip(tips, pips):
        if tip == 4:  # Thumb
            if landmarks[tip].x < landmarks[pip].x:
                raised += 1
        else:
            if landmarks[tip].y < landmarks[pip].y:
                raised += 1
    return raised

def handle_gestures(landmarks):
    global movement_locked, lock_position, gesture_hold_counter, stillness_counter
    
    fingers = count_raised_fingers(landmarks)
    wrist_pos = (landmarks[0].x, landmarks[0].y)
    
    # Lock movement when gesture detected
    if fingers > 0 and not movement_locked:
        movement_locked = True
        lock_position = wrist_pos
        gesture_hold_counter = 0
        stillness_counter = 0  # Reset stillness counter when gesture detected
        
        if fingers == 1:
            pyautogui.click()
            return "Click"
        elif fingers == 2:
            pyautogui.doubleClick()
            return "Double Click"
        elif fingers == 3:
            pyautogui.scroll(100)
            return "Scroll Up"
        elif fingers == 4:
            pyautogui.scroll(-100)
            return "Scroll Down"
    
    # Check if should unlock
    if movement_locked:
        if fingers == 0:  # No fingers raised
            gesture_hold_counter += 1
            if gesture_hold_counter >= GESTURE_HOLD_THRESHOLD:
                movement_locked = False
                lock_position = None
                return "Unlocked"
        else:  # Still gesturing
            gesture_hold_counter = 0
    
    return "Locked" if movement_locked else "Ready"

def move_cursor(landmarks):
    global last_position, movement_locked, stillness_counter, last_movement
    
    if not landmarks:
        return
    
    current_x, current_y = landmarks[0].x, landmarks[0].y
    
    if last_position:
        # Calculate immediate movement
        dx = current_x - last_position[0]
        dy = current_y - last_position[1]
        movement_magnitude = math.sqrt(dx**2 + dy**2)
        last_movement = movement_magnitude
        
        # Check for minimal movement (potential stillness)
        if movement_magnitude < STILLNESS_THRESHOLD:
            stillness_counter += 1
            if stillness_counter >= STILLNESS_FRAMES and not movement_locked:
                movement_locked = True
        else:
            stillness_counter = 0
            movement_locked = False
        
        # Only move cursor if not locked and movement is significant
        if not movement_locked and movement_magnitude >= MIN_MOVEMENT:
            # Convert to screen coordinates and apply scaling
            screen_w, screen_h = pyautogui.size()
            target_x, target_y = pyautogui.position()
            
            # Calculate new target position
            target_x += dx * screen_w * MOVEMENT_SCALE
            target_y += dy * screen_h * MOVEMENT_SCALE
            
            # Apply screen boundaries
            target_x = max(0, min(screen_w-1, target_x))
            target_y = max(0, min(screen_h-1, target_y))
            
            # Smooth movement
            smoothed_x = target_x  # Removed smoothing for immediate response
            smoothed_y = target_y
            
            try:
                pyautogui.moveTo(int(smoothed_x), int(smoothed_y))
            except:
                pass
    
    last_position = (current_x, current_y)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    status = "No Hands"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            status = handle_gestures(hand_landmarks.landmark)
            move_cursor(hand_landmarks.landmark)
            
            # Visual feedback
            wrist = hand_landmarks.landmark[0]
            cx, cy = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])
            color = (0, 0, 255) if movement_locked else (0, 255, 0)
            cv2.circle(frame, (cx, cy), 10, color, -1)
            
            fingers = count_raised_fingers(hand_landmarks.landmark)
            cv2.putText(frame, f"Fingers: {fingers}", (cx+15, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display status
    cv2.putText(frame, f"Status: {status}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Movement: {'LOCKED' if movement_locked else 'ACTIVE'}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Stillness: {stillness_counter}/{STILLNESS_FRAMES}", (10, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()                                                                      
