import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
try:
    mp_face_mesh_module = mp.solutions.face_mesh
except AttributeError:
    mp_face_mesh_module = None


# ─────────────────────────────────────────────
#  SYMMETRY THRESHOLDS
# ─────────────────────────────────────────────
#
#  Static asymmetry normal range: 0.05 to 0.15
#  Source: Grammer & Thornhill (1994), Rhodes (2006)
#  Below 0.05 = unnaturally symmetric (suspicious)
#  Above 0.15 = noticeably asymmetric (suspicious)
#
#  Asymmetry CV normal range: 0.10 to 0.60
#  Same logic as blink consistency CV
#  Low CV = face geometry is robotically stable
#  High CV = face geometry fluctuates erratically
#
#  Motion symmetry threshold: 0.30
#  Both sides should move within 30% of each other
#  Above 0.30 difference = sides moving independently
#
#  Regional asymmetry: same 0.05 to 0.15 range
#  Applied independently to upper, middle, lower zones

STATIC_ASYM_LOW    = 0.05   # below = unnaturally symmetric
STATIC_ASYM_HIGH   = 0.15   # above = noticeably asymmetric
ASYM_CV_LOW        = 0.10   # below = suspiciously stable
ASYM_CV_HIGH       = 0.60   # above = suspiciously erratic
MOTION_ASYM_THRESH = 0.30   # above = sides moving independently


# ─────────────────────────────────────────────
#  LANDMARK INDICES
# ─────────────────────────────────────────────

# Mirrored pairs for static and motion symmetry
# Format: (left_index, right_index, region_name)
SYMMETRY_PAIRS = [
    (234, 454, "cheek"),
    (130, 359, "outer eye"),
    (133, 362, "inner eye"),
    (70,  300, "eyebrow"),
    (61,  291, "mouth corner"),
    (172, 397, "jaw"),
]

# Regional zones — left and right landmark groups
UPPER_ZONE_LEFT  = [70, 63, 105, 66, 107]    # left eyebrow
UPPER_ZONE_RIGHT = [300, 293, 334, 296, 336]  # right eyebrow

MIDDLE_ZONE_LEFT  = [33, 133, 159, 145]       # left eye region
MIDDLE_ZONE_RIGHT = [263, 362, 386, 374]      # right eye region

LOWER_ZONE_LEFT  = [61, 84, 181, 91, 146]    # left mouth and lip
LOWER_ZONE_RIGHT = [291, 314, 405, 321, 375]  # right mouth and lip

# Nose tip as center reference
NOSE_TIP = 4

# Zone colors for visualization (BGR)
UPPER_COLOR  = (255, 200, 0)    # yellow
MIDDLE_COLOR = (0, 200, 255)    # cyan
LOWER_COLOR  = (200, 0, 255)    # purple
CENTER_COLOR = (0, 255, 0)      # green


# ─────────────────────────────────────────────
#  STATIC ASYMMETRY
# ─────────────────────────────────────────────

def calculate_static_asymmetry(landmarks):
    """
    Measures facial asymmetry by comparing mirrored landmark pairs
    against the nose tip center line.

    For each pair:
        left_dist  = horizontal distance of left point from center
        right_dist = horizontal distance of right point from center

        asymmetry = |left_dist - right_dist| / (left_dist + right_dist)

    Returns mean asymmetry across all pairs (0 = perfect symmetry,
    1 = completely asymmetric). Normal range: 0.05 to 0.15
    """
    center_x = landmarks[NOSE_TIP][0]
    scores = []

    for left_idx, right_idx, _ in SYMMETRY_PAIRS:
        left_dist  = abs(landmarks[left_idx][0]  - center_x)
        right_dist = abs(landmarks[right_idx][0] - center_x)

        total = left_dist + right_dist
        if total > 0:
            asym = abs(left_dist - right_dist) / total
            scores.append(asym)

    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────
#  REGIONAL SYMMETRY
# ─────────────────────────────────────────────

def calculate_regional_asymmetry(landmarks, left_indices, right_indices):
    """
    Measures asymmetry within a specific facial zone by comparing
    mean horizontal positions of left and right landmark groups
    against the nose tip center line.

    Returns asymmetry score for this zone (0 to 1).
    """
    center_x = landmarks[NOSE_TIP][0]

    left_dists  = [abs(landmarks[i][0] - center_x) for i in left_indices]
    right_dists = [abs(landmarks[i][0] - center_x) for i in right_indices]

    mean_left  = float(np.mean(left_dists))
    mean_right = float(np.mean(right_dists))

    total = mean_left + mean_right
    if total == 0:
        return 0.0

    return abs(mean_left - mean_right) / total


# ─────────────────────────────────────────────
#  MOTION SYMMETRY
# ─────────────────────────────────────────────

def calculate_motion_asymmetry(landmarks_current, landmarks_previous):
    """
    Measures whether both sides of the face moved equally between frames.

    Computes mean displacement of left-side landmarks and right-side
    landmarks separately, then compares them.

    motion_asymmetry = |left_motion - right_motion| /
                       (left_motion + right_motion)

    High value = sides moving independently = suspicious
    """
    left_indices  = [p[0] for p in SYMMETRY_PAIRS]
    right_indices = [p[1] for p in SYMMETRY_PAIRS]

    # Left side displacement
    left_curr = landmarks_current[left_indices]
    left_prev = landmarks_previous[left_indices]
    left_disp = float(np.mean(np.sqrt(((left_curr - left_prev)**2).sum(axis=1))))

    # Right side displacement
    right_curr = landmarks_current[right_indices]
    right_prev = landmarks_previous[right_indices]
    right_disp = float(np.mean(np.sqrt(((right_curr - right_prev)**2).sum(axis=1))))

    total = left_disp + right_disp
    if total == 0:
        return 0.0

    return abs(left_disp - right_disp) / total


# ─────────────────────────────────────────────
#  FULL SYMMETRY ANALYSIS
# ─────────────────────────────────────────────

def analyze_symmetry(landmarks_history):
    """
    Runs complete symmetry analysis across all frames:

    1. Static asymmetry per frame + mean + CV
    2. Regional asymmetry — upper, middle, lower zones
    3. Motion symmetry — bilateral coordination across frames

    Returns all scores and status flags.
    """

    if len(landmarks_history) < 2:
        return {"status": "INSUFFICIENT DATA"}

    static_scores  = []
    upper_scores   = []
    middle_scores  = []
    lower_scores   = []
    motion_scores  = []

    for i, landmarks in enumerate(landmarks_history):

        # Static asymmetry this frame
        static_scores.append(calculate_static_asymmetry(landmarks))

        # Regional asymmetry this frame
        upper_scores.append(calculate_regional_asymmetry(
            landmarks, UPPER_ZONE_LEFT, UPPER_ZONE_RIGHT))
        middle_scores.append(calculate_regional_asymmetry(
            landmarks, MIDDLE_ZONE_LEFT, MIDDLE_ZONE_RIGHT))
        lower_scores.append(calculate_regional_asymmetry(
            landmarks, LOWER_ZONE_LEFT, LOWER_ZONE_RIGHT))

        # Motion symmetry — needs previous frame
        if i > 0:
            motion_scores.append(calculate_motion_asymmetry(
                landmarks_history[i],
                landmarks_history[i - 1]
            ))

    # ── Static asymmetry stats ──
    mean_static = float(np.mean(static_scores))
    std_static  = float(np.std(static_scores))
    cv_static   = round(std_static / mean_static, 4) if mean_static > 0 else 0

    if mean_static < STATIC_ASYM_LOW:
        static_flag = "SUSPICIOUSLY SYMMETRIC"
    elif mean_static <= STATIC_ASYM_HIGH:
        static_flag = "NORMAL"
    else:
        static_flag = "SUSPICIOUSLY ASYMMETRIC"

    if cv_static < ASYM_CV_LOW:
        cv_flag = "SUSPICIOUSLY STABLE"
    elif cv_static <= ASYM_CV_HIGH:
        cv_flag = "NORMAL"
    else:
        cv_flag = "SUSPICIOUSLY ERRATIC"

    # ── Regional asymmetry stats ──
    mean_upper  = float(np.mean(upper_scores))
    mean_middle = float(np.mean(middle_scores))
    mean_lower  = float(np.mean(lower_scores))

    def region_flag(score):
        if score < STATIC_ASYM_LOW:
            return "SUSPICIOUSLY SYMMETRIC"
        elif score <= STATIC_ASYM_HIGH:
            return "NORMAL"
        else:
            return "SUSPICIOUSLY ASYMMETRIC"

    # ── Motion symmetry stats ──
    mean_motion = float(np.mean(motion_scores)) if motion_scores else 0.0
    motion_flag = "SUSPICIOUS" if mean_motion > MOTION_ASYM_THRESH else "NORMAL"

    # ── Suspicion scores (0 to 1) ──
    # Static asymmetry score
    if mean_static < STATIC_ASYM_LOW:
        static_score = round((STATIC_ASYM_LOW - mean_static) / STATIC_ASYM_LOW, 3)
    elif mean_static > STATIC_ASYM_HIGH:
        static_score = round(min((mean_static - STATIC_ASYM_HIGH) / STATIC_ASYM_HIGH, 1.0), 3)
    else:
        static_score = 0.0

    # CV score
    if cv_static < ASYM_CV_LOW:
        cv_score = round((ASYM_CV_LOW - cv_static) / ASYM_CV_LOW, 3)
    elif cv_static > ASYM_CV_HIGH:
        cv_score = round(min((cv_static - ASYM_CV_HIGH) / ASYM_CV_HIGH, 1.0), 3)
    else:
        cv_score = 0.0

    # Motion score
    motion_score = round(
        min(mean_motion / MOTION_ASYM_THRESH, 1.0), 3
    ) if mean_motion > MOTION_ASYM_THRESH else 0.0

    # Regional scores
    def region_score(val):
        if val < STATIC_ASYM_LOW:
            return round((STATIC_ASYM_LOW - val) / STATIC_ASYM_LOW, 3)
        elif val > STATIC_ASYM_HIGH:
            return round(min((val - STATIC_ASYM_HIGH) / STATIC_ASYM_HIGH, 1.0), 3)
        return 0.0

    upper_score  = region_score(mean_upper)
    middle_score = region_score(mean_middle)
    lower_score  = region_score(mean_lower)

    # Combined regional score — mean of three zones
    regional_combined = round((upper_score + middle_score + lower_score) / 3.0, 3)

    return {
        "static_scores":       static_scores,
        "mean_static":         round(mean_static, 4),
        "std_static":          round(std_static, 4),
        "cv_static":           cv_static,
        "static_flag":         static_flag,
        "cv_flag":             cv_flag,
        "static_score":        static_score,
        "cv_score":            cv_score,
        "mean_upper":          round(mean_upper, 4),
        "mean_middle":         round(mean_middle, 4),
        "mean_lower":          round(mean_lower, 4),
        "upper_flag":          region_flag(mean_upper),
        "middle_flag":         region_flag(mean_middle),
        "lower_flag":          region_flag(mean_lower),
        "upper_score":         upper_score,
        "middle_score":        middle_score,
        "lower_score":         lower_score,
        "regional_combined":   regional_combined,
        "mean_motion":         round(mean_motion, 4),
        "motion_flag":         motion_flag,
        "motion_score":        motion_score,
    }


# ─────────────────────────────────────────────
#  VISUALIZATION — LIVE OVERLAY
# ─────────────────────────────────────────────

def draw_symmetry_overlay(frame, landmarks, asym_score):
    """
    Draws symmetry zones on the live video frame.
    - Yellow dots = upper zone (eyebrows)
    - Cyan dots   = middle zone (eyes)
    - Purple dots = lower zone (mouth/lips)
    - Green line  = center symmetry axis
    - Score displayed on screen
    """
    h, w = frame.shape[:2]
    center_x = int(landmarks[NOSE_TIP][0])

    # Draw center symmetry axis
    cv2.line(frame, (center_x, 0), (center_x, h),
             CENTER_COLOR, 1)

    # Draw upper zone landmarks
    for idx in UPPER_ZONE_LEFT + UPPER_ZONE_RIGHT:
        pt = (int(landmarks[idx][0]), int(landmarks[idx][1]))
        cv2.circle(frame, pt, 3, UPPER_COLOR, -1)

    # Draw middle zone landmarks
    for idx in MIDDLE_ZONE_LEFT + MIDDLE_ZONE_RIGHT:
        pt = (int(landmarks[idx][0]), int(landmarks[idx][1]))
        cv2.circle(frame, pt, 3, MIDDLE_COLOR, -1)

    # Draw lower zone landmarks
    for idx in LOWER_ZONE_LEFT + LOWER_ZONE_RIGHT:
        pt = (int(landmarks[idx][0]), int(landmarks[idx][1]))
        cv2.circle(frame, pt, 3, LOWER_COLOR, -1)

    # Info box
    cv2.rectangle(frame, (20, 15), (320, 110), (0, 0, 0), -1)
    cv2.rectangle(frame, (20, 15), (320, 110), (255, 255, 255), 1)

    cv2.putText(frame, f"Asymmetry: {round(asym_score, 3)}",
                (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, "Yellow=Brow  Cyan=Eye  Purple=Lip",
                (30, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, "Green line = symmetry axis",
                (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


# ─────────────────────────────────────────────
#  VISUALIZATION — GRAPHS
# ─────────────────────────────────────────────

def plot_symmetry(symmetry_stats, fps):
    """
    Plots static asymmetry score over time with normal range marked.
    """
    scores = symmetry_stats["static_scores"]
    times  = [i / fps for i in range(len(scores))]

    plt.figure(figsize=(14, 5))
    plt.plot(times, scores, color="#534AB7", linewidth=1.2,
             label="Asymmetry score")

    plt.axhline(y=STATIC_ASYM_HIGH, color="red", linestyle="--",
                linewidth=1.2, label=f"Upper bound ({STATIC_ASYM_HIGH})")
    plt.axhline(y=STATIC_ASYM_LOW, color="orange", linestyle="--",
                linewidth=1.0, label=f"Lower bound ({STATIC_ASYM_LOW})")

    plt.fill_between(times, STATIC_ASYM_LOW, STATIC_ASYM_HIGH,
                     alpha=0.08, color="green", label="Normal range")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Asymmetry score")
    plt.title("Facial Asymmetry Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────

def print_symmetry_report(symmetry_stats):

    print("\n" + "=" * 55)
    print("   FACIAL SYMMETRY ANALYSIS REPORT")
    print("=" * 55)

    print(f"\n  THRESHOLDS")
    print("-" * 55)
    print(f"  Normal asymmetry    : {STATIC_ASYM_LOW} \u2013 {STATIC_ASYM_HIGH}")
    print(f"  CV normal range     : {ASYM_CV_LOW} \u2013 {ASYM_CV_HIGH}")
    print(f"  Motion threshold    : {MOTION_ASYM_THRESH}")

    print(f"\n  STATIC ASYMMETRY")
    print("-" * 55)
    print(f"  Mean asymmetry      : {symmetry_stats['mean_static']}  (normal: {STATIC_ASYM_LOW}\u2013{STATIC_ASYM_HIGH})")
    print(f"  Std deviation       : {symmetry_stats['std_static']}")
    print(f"  Asymmetry CV        : {symmetry_stats['cv_static']}  (normal: {ASYM_CV_LOW}\u2013{ASYM_CV_HIGH})")
    print(f"  Geometry status     : {symmetry_stats['static_flag']}")
    print(f"  Consistency status  : {symmetry_stats['cv_flag']}")
    print(f"  Static score        : {symmetry_stats['static_score']}")
    print(f"  CV score            : {symmetry_stats['cv_score']}")

    print(f"\n  REGIONAL SYMMETRY")
    print("-" * 55)
    print(f"  Upper zone (brows)  : {symmetry_stats['mean_upper']}  \u2192  {symmetry_stats['upper_flag']}  (score: {symmetry_stats['upper_score']})")
    print(f"  Middle zone (eyes)  : {symmetry_stats['mean_middle']}  \u2192  {symmetry_stats['middle_flag']}  (score: {symmetry_stats['middle_score']})")
    print(f"  Lower zone (lips)   : {symmetry_stats['mean_lower']}  \u2192  {symmetry_stats['lower_flag']}  (score: {symmetry_stats['lower_score']})")
    print(f"  Regional combined   : {symmetry_stats['regional_combined']}")

    print(f"\n  MOTION SYMMETRY")
    print("-" * 55)
    print(f"  Mean motion asym    : {symmetry_stats['mean_motion']}  (threshold: {MOTION_ASYM_THRESH})")
    print(f"  Motion status       : {symmetry_stats['motion_flag']}")
    print(f"  Motion score        : {symmetry_stats['motion_score']}")

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
    print(f"[+] Processing symmetry analysis... (press ESC to stop)\n")

    landmarks_history = []
    frame_index       = 0

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

            current_asym = 0.0

            if results.multi_face_landmarks:
                face_lms = results.multi_face_landmarks[0]

                coords = np.array(
                    [(lm.x * w, lm.y * h) for lm in face_lms.landmark],
                    dtype=np.float32
                )

                landmarks_history.append(coords)

                # Calculate live asymmetry for display
                current_asym = calculate_static_asymmetry(coords)

                # Draw symmetry overlay
                draw_symmetry_overlay(frame, coords, current_asym)

            cv2.imshow("Symmetry Analysis", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_index += 1

    video.release()
    cv2.destroyAllWindows()

    if not landmarks_history:
        print("[ERROR] No face detected in video.")
        return

    # Run full symmetry analysis
    symmetry_stats = analyze_symmetry(landmarks_history)

    # Print report
    print_symmetry_report(symmetry_stats)

    # Plot asymmetry over time
    plot_symmetry(symmetry_stats, fps)

    return symmetry_stats


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python symmetry_analysis.py <video_path>")
        print("Example: python symmetry_analysis.py video1.mp4")
        sys.exit(1)

    process_video(sys.argv[1])