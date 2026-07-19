import os
import threading

from deepface import DeepFace
import cv2
from pydub import AudioSegment
import simpleaudio as sa
import random

# --- Secure Absolute Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_absolute_track(relative_path):
    return os.path.join(BASE_DIR, *relative_path.split('/'))

# --- Audio State Controller ---
class AudioController:
    def __init__(self):
        self.play_handle = None
        self.playing_path = None

    def play(self, target_path, fade_in=1500, fade_out=2000):
        # If this exact song is already running, leave it alone
        if self.playing_path == target_path and self.is_playing():
            return

        # Interrupt active playback if it exists
        self.stop()

        try:
            print(f"Switching audio track to: {target_path}")
            sound = AudioSegment.from_file(target_path)
            
            # Apply processing rules
            if len(sound) > (fade_in + fade_out):
                sound = sound.fade_in(fade_in).fade_out(fade_out)
            
            # Extract raw audio arrays for simpleaudio execution
            raw_data = sound.raw_data
            
            # Trigger non-blocking native playback stream
            self.play_handle = sa.play_buffer(
                raw_data,
                num_channels=sound.channels,
                bytes_per_sample=sound.sample_width,
                sample_rate=sound.frame_rate
            )
            self.playing_path = target_path
        except Exception as e:
            print(f"Playback initiation failure: {e}")

    def stop(self):
        if self.play_handle and self.play_handle.is_playing():
            self.play_handle.stop()
        self.play_handle = None
        self.playing_path = None

    def is_playing(self):
        return self.play_handle is not None and self.play_handle.is_playing()

# Initialize Controller Object
audio_sys = AudioController()

# --- Setup System Capture Interfaces ---
cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
vc = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Persist state definitions outside the frame interval blocks
current_emotion = "No face detected"
current_gender = "Unknown"
last_triggered_track = None

index = 0
rng = random.random() # Initialize base random scalar

while vc.isOpened():
    ret, frame = vc.read()
    if not ret:
        print("Empty video buffer capture.")
        break

    index += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # ML pipeline runs every 10 frames to protect application overhead
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
                
                new_emotion = analysis[0]['dominant_emotion']
                new_gender = analysis[0]['dominant_gender']
                
                # If the emotion physically changed, recalculate the RNG choice matrix!
                if new_emotion != current_emotion:
                    rng = random.random()
                    print(f"Emotion shift detected! New RNG generated: {rng:.2f}")

                current_emotion = new_emotion
                current_gender = new_gender
                
            except Exception as e:
                print(f"DeepFace processing block error: {e}")
    else:
        if current_emotion != "No face detected":
            current_emotion = "No face detected"
            current_gender = "Unknown"
            audio_sys.stop() # Turn off sound if user walks away

    # --- Persistent Evaluation Matrix ---
    if current_gender == 'Man':
        chosen_track = None

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

        if chosen_track:
            full_track_path = get_absolute_track(chosen_track)
            
            # If the song has ended naturally, reset RNG to loop a potentially different track
            if not audio_sys.is_playing() and audio_sys.playing_path is not None:
                rng = random.random()
                print(f"Track finished naturally. Re-rolling RNG: {rng:.2f}")
            
            # Submit to state engine
            audio_sys.play(full_track_path)

    # --- UI Rendering Layer ---
    status = f"Emotion: {current_emotion} | Gender: {current_gender}"
    if audio_sys.is_playing():
        status += f" | Track: {os.path.basename(audio_sys.playing_path)}"
        
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(20) & 0xFF == 27: # ESC hook
        break

# Clean terminations
audio_sys.stop()
vc.release()
cv2.destroyAllWindows()