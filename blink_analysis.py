import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Mediapipe 0.10.13 compatible import
import mediapipe as mp
try:
    mp_face_mesh_module = mp.solutions.face_mesh
except AttributeError:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh_module


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

EAR_THRESHOLD   = 0.20   # Soukupova & Cech (2016)
CONSEC_FRAMES   = 2      # Consecutive frames below threshold to register blink

# Eye landmark indices (MediaPipe FaceMesh)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Normal ranges from literature
NORMAL_BLINK_RATE_MIN     = 8    # blinks/min — Doughty (2001)
NORMAL_BLINK_RATE_MAX     = 24
NORMAL_BLINK_DURATION_MIN = 100  # ms
NORMAL_BLINK_DURATION_MAX = 400

# Minimum video length for reliable analysis
MIN_VIDEO_DURATION_SEC = 30


# ─────────────────────────────────────────────
#  EAR CALCULATION
# ─────────────────────────────────────────────

def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def calculate_EAR(landmarks, eye_indices):
    """
    Eye Aspect Ratio — Soukupova & Cech (2016)
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical1  = euclidean(p2, p6)
    vertical2  = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    if horizontal == 0:
        return 0.0

    return round((vertical1 + vertical2) / (2.0 * horizontal), 4)


# ─────────────────────────────────────────────
#  EYE LANDMARK VISUALIZATION
# ─────────────────────────────────────────────

def draw_eye_landmarks(frame, landmarks, eye_indices, color):
    """
    Draws the 6 landmark points for one eye and connects them.
    Green = eye open, Red = eye closing/blinking
    """
    points = [landmarks[i] for i in eye_indices]

    # Draw each landmark point as a filled circle
    for point in points:
        cv2.circle(frame, point, 3, color, -1)

    # Connect points to outline the eye shape
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)


def draw_overlay(frame, ear, blink_count, is_blinking, confidence_warning):
    """
    Draws all text overlays on the frame.
    """
    h, w = frame.shape[:2]

    # Background box for cleaner text display
    cv2.rectangle(frame, (30, 20), (320, 130), (0, 0, 0), -1)
    cv2.rectangle(frame, (30, 20), (320, 130), (255, 255, 255), 1)

    # Blink count
    cv2.putText(frame, f"Blinks: {blink_count}",
                (45, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # EAR value — red when blinking, green when open
    ear_color = (0, 0, 255) if is_blinking else (0, 255, 0)
    cv2.putText(frame, f"EAR: {ear}",
                (45, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ear_color, 2)

    # Blink flash indicator
    if is_blinking:
        cv2.putText(frame, "BLINK!",
                    (45, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Low confidence warning at bottom
    if confidence_warning:
        cv2.putText(frame, "LOW CONFIDENCE: Video too short",
                    (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 165, 255), 2)


# ─────────────────────────────────────────────
#  BLINK CONSISTENCY ANALYSIS
# ─────────────────────────────────────────────

def analyze_consistency(blink_intervals):
    """
    Coefficient of Variation (CV) of blink intervals.
    CV = std / mean
        < 0.10    = suspiciously regular (robotic)
        0.10-0.60 = natural human variation
        > 0.60    = suspiciously erratic
    """
    if len(blink_intervals) < 2:
        return {"cv": None, "consistency_flag": "INSUFFICIENT DATA"}

    mean = float(np.mean(blink_intervals))
    std  = float(np.std(blink_intervals))
    cv   = round(std / mean, 4) if mean > 0 else 0

    if cv < 0.10:
        flag = "SUSPICIOUSLY REGULAR"
    elif cv <= 0.60:
        flag = "NORMAL"
    else:
        flag = "SUSPICIOUSLY IRREGULAR"

    return {
        "cv":                cv,
        "mean_interval_sec": round(mean, 3),
        "std_interval_sec":  round(std, 3),
        "consistency_flag":  flag
    }


# ─────────────────────────────────────────────
#  EAR GRAPH
# ─────────────────────────────────────────────

def plot_ear(ear_values, blink_frames, fps):
    times = [i / fps for i in range(len(ear_values))]

    plt.figure(figsize=(14, 5))
    plt.plot(times, ear_values, color="#2E75B6", linewidth=1.2, label="EAR")
    plt.axhline(y=EAR_THRESHOLD, color="red", linestyle="--",
                linewidth=1.2, label=f"Blink Threshold ({EAR_THRESHOLD})")

    for bf in blink_frames:
        plt.axvline(x=bf / fps, color="orange", linewidth=0.8, alpha=0.5)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Eye Aspect Ratio (EAR)")
    plt.title("EAR Over Time — Orange Lines = Detected Blinks")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────

def print_report(video_info, blink_count, blink_rate,
                 duration_analysis, consistency):

    if blink_rate < NORMAL_BLINK_RATE_MIN:
        rate_flag = "SUSPICIOUSLY LOW"
    elif blink_rate > NORMAL_BLINK_RATE_MAX:
        rate_flag = "SUSPICIOUSLY HIGH"
    else:
        rate_flag = "NORMAL"

    print("\n" + "=" * 55)
    print("   DEEPFAKE DETECTION — BLINK ANALYSIS REPORT")
    print("=" * 55)

    print(f"\n  VIDEO INFO")
    print("-" * 55)
    print(f"  FPS            : {video_info['fps']}")
    print(f"  Total Frames   : {video_info['total_frames']}")
    print(f"  Duration       : {video_info['duration_sec']}s")
    if video_info['duration_sec'] < MIN_VIDEO_DURATION_SEC:
        print(f"  WARNING        : Below minimum {MIN_VIDEO_DURATION_SEC}s — LOW CONFIDENCE")

    print(f"\n  BLINK DETECTION")
    print("-" * 55)
    print(f"  Total Blinks   : {blink_count}")
    print(f"  Blink Rate     : {blink_rate} blinks/min")
    print(f"  Normal Range   : {NORMAL_BLINK_RATE_MIN}-{NORMAL_BLINK_RATE_MAX} blinks/min")
    print(f"  Rate Status    : {rate_flag}")

    print(f"\n  BLINK DURATION")
    print("-" * 55)
    print(f"  Avg Duration   : {duration_analysis['avg_ms']} ms")
    print(f"  Normal Range   : {NORMAL_BLINK_DURATION_MIN}-{NORMAL_BLINK_DURATION_MAX} ms")
    print(f"  Duration Status: {duration_analysis['flag']}")

    print(f"\n  BLINK CONSISTENCY")
    print("-" * 55)
    if consistency['cv'] is not None:
        print(f"  Mean Interval  : {consistency['mean_interval_sec']}s between blinks")
        print(f"  Std Deviation  : {consistency['std_interval_sec']}s")
        print(f"  CV             : {consistency['cv']}  (normal: 0.10-0.60)")
    print(f"  Consistency    : {consistency['consistency_flag']}")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def process_video(video_path):

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps          = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = round(total_frames / fps, 2) if fps > 0 else 0

    print(f"\n[+] Video loaded: {total_frames} frames @ {fps} fps ({duration_sec}s)")

    if duration_sec < MIN_VIDEO_DURATION_SEC:
        print(f"[!] WARNING: Video is {duration_sec}s — below minimum {MIN_VIDEO_DURATION_SEC}s")
        print(f"[!] Results will be low confidence\n")
    else:
        print(f"[+] Processing... (press ESC to stop early)\n")

    # Tracking variables
    blink_count      = 0
    frame_counter    = 0
    is_blinking      = False
    ear_values       = []
    blink_frames     = []
    blink_durations  = []
    blink_timestamps = []
    frame_index      = 0
    confidence_warn  = duration_sec < MIN_VIDEO_DURATION_SEC

    with mp_face_mesh_module.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = video.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results  = face_mesh.process(rgb)

            ear = ear_values[-1] if ear_values else 0.30

            if results.multi_face_landmarks:
                face_lms  = results.multi_face_landmarks[0]
                landmarks = [
                    (int(lm.x * w), int(lm.y * h))
                    for lm in face_lms.landmark
                ]

                # EAR calculation — both eyes averaged
                left_ear  = calculate_EAR(landmarks, LEFT_EYE)
                right_ear = calculate_EAR(landmarks, RIGHT_EYE)
                ear       = round((left_ear + right_ear) / 2.0, 4)

                # Draw eye landmarks — red when blinking, green when open
                eye_color = (0, 0, 255) if ear < EAR_THRESHOLD else (0, 255, 0)
                draw_eye_landmarks(frame, landmarks, LEFT_EYE,  eye_color)
                draw_eye_landmarks(frame, landmarks, RIGHT_EYE, eye_color)

            ear_values.append(ear)

            # Blink detection — consecutive frames logic
            if ear < EAR_THRESHOLD:
                frame_counter += 1
                is_blinking = True
            else:
                if frame_counter >= CONSEC_FRAMES:
                    blink_count += 1
                    blink_frames.append(frame_index)
                    duration_ms = round((frame_counter / fps) * 1000, 1)
                    blink_durations.append(duration_ms)
                    blink_timestamps.append(round(frame_index / fps, 3))
                frame_counter = 0
                is_blinking   = False

            # Draw all overlays
            draw_overlay(frame, ear, blink_count, is_blinking, confidence_warn)

            cv2.imshow("Deepfake Detection — Blink Analysis", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_index += 1

    video.release()
    cv2.destroyAllWindows()

    # ── Final calculations ──
    duration_min = duration_sec / 60.0
    blink_rate   = round(blink_count / duration_min, 2) if duration_min > 0 else 0

    if blink_durations:
        avg_ms = round(float(np.mean(blink_durations)), 2)
        if avg_ms < NORMAL_BLINK_DURATION_MIN:
            dur_flag = "ABNORMALLY SHORT"
        elif avg_ms > NORMAL_BLINK_DURATION_MAX:
            dur_flag = "ABNORMALLY LONG"
        else:
            dur_flag = "NORMAL"
    else:
        avg_ms   = 0
        dur_flag = "NO BLINKS DETECTED"

    duration_analysis = {"avg_ms": avg_ms, "flag": dur_flag}

    blink_intervals = [
        round(blink_timestamps[i] - blink_timestamps[i-1], 3)
        for i in range(1, len(blink_timestamps))
    ]
    consistency = analyze_consistency(blink_intervals)

    video_info = {
        "fps":          round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": duration_sec
    }

    print_report(video_info, blink_count, blink_rate, duration_analysis, consistency)
    plot_ear(ear_values, blink_frames, fps)


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python blink_analysis_final.py <video_path>")
        print("Example: python blink_analysis_final.py video1.mp4")
        sys.exit(1)

    process_video(sys.argv[1])