import os
import random
import cv2
from deepface import DeepFace
import pygame
from collections import deque, Counter

# --- Secure Absolute Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_absolute_track(relative_path):
    return os.path.join(BASE_DIR, *relative_path.split('/'))

# --- Audio State Controller ---
class AudioController:
    def __init__(self):
        pygame.mixer.init()
        self.channel1 = pygame.mixer.Channel(0)
        self.channel2 = pygame.mixer.Channel(1)
        self.active_channel = self.channel1
        self.playing_path = ""
    
    def play(self, target_path, fade_in=1500, fade_out=1500):
        if self.playing_path == target_path and self.active_channel.get_busy():
            return

        if self.active_channel.get_busy():
            self.active_channel.fadeout(fade_out)
        
        self.active_channel = self.channel2 if self.active_channel == self.channel1 else self.channel1

        try:
            print(f"Switching audio track to: {target_path}")
            sound = pygame.mixer.Sound(target_path)
            self.active_channel.play(sound, fade_ms=fade_in)
            self.playing_path = target_path
        except Exception as e:
            print(f"Playback initiation failure: {e}")
            self.playing_path = ""

    def stop(self, fade_out=1500):
        if self.active_channel.get_busy():
            self.active_channel.fadeout(fade_out)
        self.playing_path = ""

    def is_playing(self):
        return self.active_channel.get_busy()

audio_sys = AudioController()

# --- Setup System Capture Interfaces ---
cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
vc = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- STABILITY TRACKERS ---
current_emotion = "No face detected"
current_gender = "Unknown"

# Rolling buffers for ML readings (stores the last 5 readings)
emotion_buffer = deque(maxlen=5)
gender_buffer = deque(maxlen=5)

# Grace period counters for face detection (prevents audio cutting if you blink or turn your head)
frames_without_face = 0
MAX_FACELESS_FRAMES = 30 # roughly 1 second at 30fps

index = 0
rng = random.random()

while vc.isOpened():
    ret, frame = vc.read()
    if not ret:
        print("Empty video buffer capture.")
        break

    index += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        frames_without_face = 0 # Reset the grace period counter

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        if index % 10 == 0:
            try:
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                
                analysis = DeepFace.analyze(
                    img_path=face_roi, 
                    actions=['gender', 'emotion'], 
                    enforce_detection=False, 
                    silent=True
                )
                
                # Append raw readings to rolling buffers
                emotion_buffer.append(analysis[0]['dominant_emotion'])
                gender_buffer.append(analysis[0]['dominant_gender'])
                
                # Extract the most common element in the buffers
                smoothed_emotion = Counter(emotion_buffer).most_common(1)[0][0]
                smoothed_gender = Counter(gender_buffer).most_common(1)[0][0]
                
                if smoothed_emotion != current_emotion:
                    rng = random.random()
                    print(f"Emotion shift detected! ({current_emotion} -> {smoothed_emotion}) New RNG: {rng:.2f}")

                current_emotion = smoothed_emotion
                current_gender = smoothed_gender
                
            except Exception as e:
                pass # Suppress DeepFace errors
    else:
        # Increment the faceless counter instead of instantly stopping
        frames_without_face += 1
        
        if frames_without_face > MAX_FACELESS_FRAMES:
            if current_emotion != "No face detected":
                print("Face lost for >1 second. Clearing state and fading out audio.")
                current_emotion = "No face detected"
                current_gender = "Unknown"
                emotion_buffer.clear()
                gender_buffer.clear()
                audio_sys.stop() 

    # --- Persistent Evaluation Matrix ---
    chosen_track = None

    if current_gender == 'Man':
        if current_emotion == 'happy':
            if rng < 0.33:
                chosen_track = "audio/Happy/Happy1.wav"
            elif rng < 0.66:
                chosen_track = "audio/Happy/Happy2.wav"
            else:
                chosen_track = "audio/Happy/Happy3.wav"

        elif current_emotion == 'sad':
            if rng < 0.33:
                chosen_track = "audio/Sad/Sad1.wav"
            elif rng < 0.66:
                chosen_track = "audio/Sad/Sad2.wav"
            else:
                chosen_track = "audio/Sad/Sad3.wav"

        elif current_emotion == 'angry':
            chosen_track = "audio/Angry/Angry1.wav"

    # --- State Execution Engine ---
    if chosen_track:
        full_track_path = get_absolute_track(chosen_track)
        
        if not audio_sys.is_playing() and audio_sys.playing_path != "":
            rng = random.random()
            print(f"Track finished naturally. Re-rolling RNG: {rng:.2f}")
        
        audio_sys.play(full_track_path)
    else:
        # Stop existing audio if an unmapped emotion (like fear/neutral) is stable enough to become the current_emotion
        if audio_sys.is_playing() and current_emotion != "No face detected":
            audio_sys.stop()

    # --- UI Rendering Layer ---
    status = f"Emotion: {current_emotion} | Gender: {current_gender}"
    if audio_sys.is_playing():
        status += f" | Track: {os.path.basename(audio_sys.playing_path)}"
        
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Optional debug text to see the buffer at work
    # cv2.putText(frame, f"Raw Buffer: {list(emotion_buffer)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(20) & 0xFF == 27:
        break

audio_sys.stop()
pygame.quit()
vc.release()
cv2.destroyAllWindows()