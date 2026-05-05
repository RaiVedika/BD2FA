import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
mp_face_mesh_module = mp.solutions.face_mesh


# ─────────────────────────────────────────────
#  BIOMECHANICAL THRESHOLDS
# ─────────────────────────────────────────────
#
#  All thresholds derived from biomechanical constraints
#  and validated through real video testing.
#
#  JUMP threshold
#  Basis: orbicularis oculi min contraction = 100ms
#         at 30fps = 33ms per frame = 33% of fastest movement
#         max eye opening ~30px → max displacement = face_width × 0.05
#  Refinement: requires 2 consecutive frames (not 1) to distinguish
#              natural head movement from deepfake geometric discontinuities
#              (validated from real video testing — single frame jumps
#               occur naturally during quick head movements)
#
#  FREEZE threshold
#  Basis: natural micro-movement from breathing and postural sway
#         is always present at minimum 0.5px per frame in real humans
#  Status: validated — correctly produced 0 freeze events on real video
#
#  SMOOTHNESS threshold
#  Basis: bell-shaped velocity profile of facial muscles
#         movement cannot accelerate or decelerate instantaneously
#  Status: validated — score of 0.043 on real video confirms correct sensitivity
#
#  JITTER CV bounds
#  Lower bound (0.10): suspiciously regular / robotic motion
#  Upper bound (0.85): raised from initial 0.60 after real video testing
#                      showed natural facial motion variability produces
#                      CV of ~0.755 due to behavioral variation in talking,
#                      nodding, and expression changes
#                      (empirical finding — reported in calibration document)

JUMP_FACE_MULTIPLIER   = 0.05   # jump threshold = face_width × this
JUMP_CONSEC_FRAMES     = 2      # consecutive frames required to confirm jump
FREEZE_THRESHOLD_PX    = 0.5    # below this = unnaturally frozen
FREEZE_MIN_FRAMES      = 10     # must be frozen for this many consecutive frames
SMOOTH_MULTIPLIER      = 0.5    # smoothness violation = change > jump × this
JITTER_CV_LOW          = 0.10   # below this = suspiciously smooth / robotic
JITTER_CV_HIGH         = 0.85   # above this = suspiciously erratic
                                # (calibrated from 0.60 after real video testing)


# ─────────────────────────────────────────────
#  FACE WIDTH CALCULATION
# ─────────────────────────────────────────────

def get_face_width(landmarks_array):
    """
    Calculates face width in pixels using cheek landmarks.
    Used to set biomechanically grounded jump threshold.

    MediaPipe indices:
        234 = left cheek point
        454 = right cheek point
    """
    left_cheek  = landmarks_array[234]
    right_cheek = landmarks_array[454]

    face_width = float(np.sqrt(
        (right_cheek[0] - left_cheek[0])**2 +
        (right_cheek[1] - left_cheek[1])**2
    ))
    return round(face_width, 2)


# ─────────────────────────────────────────────
#  DISPLACEMENT CALCULATION
# ─────────────────────────────────────────────

def calculate_displacement(landmarks_current, landmarks_previous):
    """
    Calculates mean Euclidean displacement of all 468 landmarks
    between two consecutive frames.

    displacement(t) = mean of ||landmark_i(t) - landmark_i(t-1)||
                      for all i in 468

    Returns a single float representing average facial movement
    in pixels between the two frames.
    """
    diff      = landmarks_current - landmarks_previous
    distances = np.sqrt((diff**2).sum(axis=1))
    return float(np.mean(distances))


# ─────────────────────────────────────────────
#  TEMPORAL ANALYSIS
# ─────────────────────────────────────────────

def analyze_temporal(landmarks_history, fps, face_width):
    """
    Analyzes frame-to-frame landmark displacement using
    calibrated biomechanical thresholds to detect:

    1. Jump events   — 2+ consecutive frames beyond physical displacement limit
    2. Freeze events — face unnaturally still for extended period
    3. Smoothness    — abrupt changes in motion violating muscle physics
    4. Jitter        — overall motion consistency using CV
    """

    if len(landmarks_history) < 2:
        return {"status": "INSUFFICIENT DATA"}

    # ── Calculate displacement for every consecutive frame pair ──
    displacements = []
    for i in range(1, len(landmarks_history)):
        d = calculate_displacement(
            landmarks_history[i],
            landmarks_history[i - 1]
        )
        displacements.append(d)

    displacements = np.array(displacements)

    # ── Biomechanical thresholds ──
    jump_threshold   = face_width * JUMP_FACE_MULTIPLIER
    smooth_threshold = jump_threshold * SMOOTH_MULTIPLIER

    # ── Jump detection ──
    # Requires JUMP_CONSEC_FRAMES consecutive frames above threshold
    # Single frame jumps = natural head movement (validated from real video)
    # Multiple consecutive frame jumps = deepfake geometric discontinuity
    jump_events  = 0
    jump_counter = 0

    for d in displacements:
        if d > jump_threshold:
            jump_counter += 1
            if jump_counter == JUMP_CONSEC_FRAMES:
                jump_events += 1
        else:
            jump_counter = 0

    jump_score = round(min(jump_events / max(len(displacements) * 0.005, 1), 1.0), 3)

    # ── Freeze detection ──
    # Face stays below micro-movement floor for 10+ consecutive frames
    freeze_events  = 0
    freeze_counter = 0

    for d in displacements:
        if d < FREEZE_THRESHOLD_PX:
            freeze_counter += 1
            if freeze_counter == FREEZE_MIN_FRAMES:
                freeze_events += 1
        else:
            freeze_counter = 0

    freeze_score = round(min(freeze_events / 3.0, 1.0), 3)

    # ── Smoothness analysis ──
    # Rate of change of displacement between consecutive frames
    # Real muscles follow bell-shaped velocity — cannot change abruptly
    changes           = np.abs(np.diff(displacements))
    smooth_violations = int(np.sum(changes > smooth_threshold))
    smooth_score      = round(
        min(smooth_violations / max(len(changes) * 0.05, 1), 1.0), 3
    )

    # ── Jitter analysis ──
    # Coefficient of variation of displacement series
    # Upper bound calibrated to 0.85 from real video testing
    mean_d = float(np.mean(displacements))
    std_d  = float(np.std(displacements))
    cv     = round(std_d / mean_d, 4) if mean_d > 0 else 0

    if cv < JITTER_CV_LOW:
        jitter_flag = "SUSPICIOUSLY SMOOTH"
    elif cv <= JITTER_CV_HIGH:
        jitter_flag = "NORMAL"
    else:
        jitter_flag = "SUSPICIOUSLY JITTERY"

    return {
        "displacements":        displacements.tolist(),
        "mean_displacement":    round(mean_d, 4),
        "std_displacement":     round(std_d, 4),
        "jitter_cv":            cv,
        "jitter_flag":          jitter_flag,
        "jump_threshold_px":    round(jump_threshold, 2),
        "jump_events":          jump_events,
        "jump_score":           jump_score,
        "freeze_threshold_px":  FREEZE_THRESHOLD_PX,
        "freeze_events":        freeze_events,
        "freeze_score":         freeze_score,
        "smooth_threshold_px":  round(smooth_threshold, 2),
        "smooth_violations":    smooth_violations,
        "smooth_score":         smooth_score,
    }


# ─────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────

def plot_displacement(displacements, jump_threshold, freeze_threshold, fps):
    """
    Plots frame-to-frame displacement over time with
    jump and freeze thresholds marked.
    """
    times = [i / fps for i in range(len(displacements))]

    plt.figure(figsize=(14, 5))
    plt.plot(times, displacements, color="#1D9E75", linewidth=1.2,
             label="Mean landmark displacement")

    plt.axhline(y=jump_threshold, color="red", linestyle="--",
                linewidth=1.2, label=f"Jump threshold ({round(jump_threshold, 1)}px)")
    plt.axhline(y=freeze_threshold, color="orange", linestyle="--",
                linewidth=1.0, label=f"Freeze threshold ({freeze_threshold}px)")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Displacement (pixels)")
    plt.title("Facial Landmark Displacement Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────

def print_temporal_report(temporal_stats, face_width):

    print("\n" + "=" * 55)
    print("   TEMPORAL ANALYSIS REPORT  (calibrated v2)")
    print("=" * 55)

    print(f"\n  BIOMECHANICAL THRESHOLDS")
    print("-" * 55)
    print(f"  Face width detected  : {face_width}px")
    print(f"  Jump threshold       : {temporal_stats['jump_threshold_px']}px  (face \u00d7 {JUMP_FACE_MULTIPLIER})")
    print(f"  Jump min frames      : {JUMP_CONSEC_FRAMES} consecutive frames")
    print(f"  Freeze threshold     : {temporal_stats['freeze_threshold_px']}px  (micro-movement floor)")
    print(f"  Smooth threshold     : {temporal_stats['smooth_threshold_px']}px  (jump \u00d7 {SMOOTH_MULTIPLIER})")
    print(f"  Jitter CV range      : {JITTER_CV_LOW} \u2013 {JITTER_CV_HIGH}  (calibrated from 0.60)")

    print(f"\n  DISPLACEMENT STATS")
    print("-" * 55)
    print(f"  Mean displacement    : {temporal_stats['mean_displacement']}px per frame")
    print(f"  Std deviation        : {temporal_stats['std_displacement']}px")
    print(f"  Jitter CV            : {temporal_stats['jitter_cv']}  (normal: {JITTER_CV_LOW}\u2013{JITTER_CV_HIGH})")
    print(f"  Jitter status        : {temporal_stats['jitter_flag']}")

    print(f"\n  ANOMALY DETECTION")
    print("-" * 55)
    print(f"  Jump events          : {temporal_stats['jump_events']}  (score: {temporal_stats['jump_score']})")
    print(f"  Freeze events        : {temporal_stats['freeze_events']}  (score: {temporal_stats['freeze_score']})")
    print(f"  Smooth violations    : {temporal_stats['smooth_violations']}  (score: {temporal_stats['smooth_score']})")

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
    print(f"[+] Processing temporal analysis... (press ESC to stop)\n")

    landmarks_history = []
    face_width        = None
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

            if results.multi_face_landmarks:
                face_lms = results.multi_face_landmarks[0]

                # Convert to pixel coordinate array shape (468, 2)
                coords = np.array(
                    [(lm.x * w, lm.y * h) for lm in face_lms.landmark],
                    dtype=np.float32
                )

                landmarks_history.append(coords)

                # Calculate face width from first detected frame
                if face_width is None:
                    face_width = get_face_width(coords)
                    print(f"[+] Face width         : {face_width}px")
                    print(f"[+] Jump threshold     : {round(face_width * JUMP_FACE_MULTIPLIER, 1)}px")
                    print(f"[+] Requires           : {JUMP_CONSEC_FRAMES} consecutive frames\n")

                # Draw all 468 landmarks on frame
                for point in coords:
                    cv2.circle(frame, (int(point[0]), int(point[1])),
                               1, (0, 255, 150), -1)

            cv2.putText(frame, f"Frame: {frame_index}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)

            cv2.imshow("Temporal Analysis", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_index += 1

    video.release()
    cv2.destroyAllWindows()

    if not landmarks_history or face_width is None:
        print("[ERROR] No face detected in video.")
        return

    # Run temporal analysis
    temporal_stats = analyze_temporal(landmarks_history, fps, face_width)

    # Print report
    print_temporal_report(temporal_stats, face_width)

    # Plot displacement graph
    plot_displacement(
        temporal_stats["displacements"],
        temporal_stats["jump_threshold_px"],
        temporal_stats["freeze_threshold_px"],
        fps
    )

    return temporal_stats


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python temporal_analysis.py <video_path>")
        print("Example: python temporal_analysis.py video1.mp4")
        sys.exit(1)

    process_video(sys.argv[1])