import cv2
import signal
import sys
import RPi.GPIO as GPIO
from time import sleep

# -------------------------------------
# Servo Motor Setup and Control Section
# -------------------------------------

servo_pin = 18  # GPIO pin connected to the servo

GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

# Initialize PWM on the servo pin at 50Hz (common for hobby servos)
pwm = GPIO.PWM(servo_pin, 50)
pwm.start(0)

# Servo movement speed (sec/60°).
SERVO_DELAY_RATE = 0.2  # seconds
# A fixed step (in degrees) to move the servo each update.
SERVO_STEP = 5.0

def set_angle(angle):
    """
    Move the servo to the specified angle.
    Converts the angle (0-180) to the corresponding duty cycle.
    """
    angle = abs(angle)
    duty = 2 + (angle / 18)  # Convert angle to duty cycle
    GPIO.output(servo_pin, True)
    pwm.ChangeDutyCycle(duty)
    sleep(SERVO_DELAY_RATE * angle / 60)  # Allow time for the servo to move
    GPIO.output(servo_pin, False)
    pwm.ChangeDutyCycle(0)  # Stop sending the signal to prevent jitter

# -------------------------------------
# Face Detection and Video Capture Setup
# -------------------------------------

# Replace with the streaming URL from your phone
stream_url = "http://192.168.0.239:4747/video"

# Load the pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global video capture object
cap = None

# -------------------------------------
# Cleanup Function and Signal Handler
# -------------------------------------

def cleanup():
    """
    Release video capture, close windows, and clean up GPIO resources.
    """
    global cap
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    pwm.stop()
    GPIO.cleanup()
    print("Cleaned up resources.")

def signal_handler(sig, frame):
    """
    Handler for the SIGINT signal.
    """
    print("\nProgram terminated by user!")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# -------------------------------------
# Main Loop: Face Tracking with Servo Control
# -------------------------------------

try:
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Cannot access the video stream")
        sys.exit(1)

    # Target width for processing (preserved aspect ratio)
    target_width = 320

    # Process every nth frame for face detection and servo update
    frame_skip = 24
    frame_count = 0
    last_face_position = None

    # Servo adjustment parameters:
    center_threshold = 10  # Minimum pixel difference to trigger a servo update

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame - trying to reconnect...")
            cap.release()
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print("Reconnection failed - check your stream source")
                break
            continue

        # Rotate the frame by 90° clockwise to correct orientation.
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Resize while preserving aspect ratio (target width = 320)
        (h, w) = rotated_frame.shape[:2]
        scale = target_width / float(w)
        new_height = int(h * scale)
        frame = cv2.resize(rotated_frame, (target_width, new_height), interpolation=cv2.INTER_AREA)
        frame_count += 1

        # ----------------------------
        # Face Detection and Servo Update Section
        # ----------------------------
        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(30, 30)
            )

            if len(faces) > 0:
                # Use the first detected face
                last_face_position = faces[0]
                (x, y, w_face, h_face) = last_face_position
                face_center_x = x + w_face // 2
                frame_center_x = frame.shape[1] // 2
                error = frame_center_x - face_center_x
                
                # If the face is left of center (error < -threshold), rotate left
                if error < -center_threshold:
                    angle = (abs(error) - center_threshold) * SERVO_STEP / (90 - center_threshold)
                    print(f"Face left of center. Rotating {angle:1f}° to the left.")
                    set_angle(angle)
            else:
                last_face_position = None

        # ----------------------------
        # Drawing the Face Detection Overlay
        # ----------------------------
        if last_face_position is not None:
            (x, y, w_face, h_face) = last_face_position
            if x >= 0 and y >= 0 and w_face > 0 and h_face > 0:
                cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (255, 0, 0), 2)
                center_x = x + w_face // 2
                center_y = y + h_face // 2
                cv2.putText(frame, f"({center_x}, {center_y})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the video feed
        cv2.imshow("Face Tracker", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    cleanup()
