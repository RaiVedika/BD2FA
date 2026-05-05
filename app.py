import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime

# ── Real analysis modules ──────────────────────────────────────────────────────
import mediapipe as mp
mp_face_mesh_module = mp.solutions.face_mesh

from blink_analysis import (
    calculate_EAR, LEFT_EYE, RIGHT_EYE,
    EAR_THRESHOLD, NORMAL_BLINK_RATE_MIN, NORMAL_BLINK_RATE_MAX,
    NORMAL_BLINK_DURATION_MIN, NORMAL_BLINK_DURATION_MAX,
    analyze_consistency
)
from symmetry_analysis import (
    calculate_static_asymmetry, analyze_symmetry
)
from temporal_analysis import (
    get_face_width, calculate_displacement, analyze_temporal,
    JUMP_FACE_MULTIPLIER, FREEZE_THRESHOLD_PX
)


# ─────────────────────────────────────────────
#  PAGE CONFIG & STYLING
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BD2FA: Behavioral-Based Deepfake Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #182c4d;
        color: #E2E8F0;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
        font-weight: 600;
    }
    .stFileUploader label {
        color: #e3edf6 !important;
        font-weight: 500;
    }
    .stFileUploader label {
        color: #e3edf6 !important;
        font-weight: 500;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #3b5a8f !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] button {
        color: #374151 !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #f3f4f6 !important;
        border: 1px solid #d1d5db !important;
    }
    .stStatus {
        background-color: #1f3555 !important;
    }
    [data-testid="metric-container"] {
        background-color: rgba(31, 53, 85, 0.8);
        border-radius: 10px;
        padding: 16px !important;
    }
    [data-testid="metric-container"] > div {
        padding: 8px !important;
    }
    [data-testid="metric-container"] label {
        color: #B8C5D6 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 32px !important;
        font-weight: 700 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #86b6ca !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    h3 { color: #FFFFFF !important; font-weight: 700 !important; font-size: 18px !important; }
    h2 { color: #FFFFFF !important; font-weight: 700 !important; font-size: 20px !important; }
    [data-testid="stAlert"] {
        background-color: rgba(31, 53, 85, 0.8);
        color: #E2E8F0;
    }
    [data-testid="stAlert"] p { color: #E2E8F0 !important; }
    p, li, span { color: #E2E8F0 !important; }
    .stDownloadButton button {
        color: #FFFFFF !important;
        background-color: #3b5a8f !important;
    }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def validate_video(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration

def get_frame_count(file_path):
    cap = cv2.VideoCapture(file_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

def get_video_fps(file_path):
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


# ─────────────────────────────────────────────
#  REAL ANALYSIS PIPELINE
# ─────────────────────────────────────────────
def clamp(x):
    return max(0.0, min(x, 1.0))


def run_analysis(video_path, feed_placeholder, progress_placeholder):
    """
    Single-pass pipeline: runs blink, temporal, and symmetry analysis
    simultaneously while showing live video feed. Returns all results
    needed for the dashboard.
    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── State ──────────────────────────────────────────────
    # Blink
    ear_values       = []
    blink_count      = 0
    frame_counter    = 0
    blink_frames     = []
    blink_durations  = []
    blink_timestamps = []
    is_blinking      = False

    # Temporal + Symmetry
    landmarks_history = []
    face_width        = None
    static_asym_values = []

    logs = []
    frame_index = 0

    with mp_face_mesh_module.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            ear = ear_values[-1] if ear_values else 0.30

            if results.multi_face_landmarks:
                face_lms = results.multi_face_landmarks[0]

                # Pixel coords for both uses
                coords = np.array(
                    [(lm.x * w, lm.y * h) for lm in face_lms.landmark],
                    dtype=np.float32
                )
                landmarks_history.append(coords)

                # Face width (for jump threshold)
                if face_width is None:
                    face_width = get_face_width(coords)

                # EAR (blink)
                lm_tuples = [(int(c[0]), int(c[1])) for c in coords]
                left_ear  = calculate_EAR(lm_tuples, LEFT_EYE)
                right_ear = calculate_EAR(lm_tuples, RIGHT_EYE)
                ear = round((left_ear + right_ear) / 2.0, 4)

                # Live static asymmetry
                static_asym_values.append(calculate_static_asymmetry(coords))

            ear_values.append(ear)

            # Blink detection
            if ear < EAR_THRESHOLD:
                frame_counter += 1
                is_blinking = True
            else:
                if frame_counter >= 2:
                    blink_count += 1
                    blink_frames.append(frame_index)
                    duration_ms = round((frame_counter / fps) * 1000, 1)
                    blink_durations.append(duration_ms)
                    blink_timestamps.append(round(frame_index / fps, 3))
                    logs.append(f"Frame {frame_index}: Blink detected ({duration_ms}ms)")
                frame_counter = 0
                is_blinking   = False

            # Live feed (show every frame, throttle Streamlit updates)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_index % 3 == 0:
                feed_placeholder.image(frame_rgb, use_container_width=True)

            pct = (frame_index + 1) / max(total_frames, 1)
            if frame_index % 5 == 0:
                progress_placeholder.markdown(f"""
                <div style="margin-top:8px;">
                    <div style="color:#B8C5D6; font-size:12px; margin-bottom:6px;">
                        Processing Progress — Frame {frame_index + 1}/{total_frames}
                    </div>
                    <div style="background:#1f3555; border-radius:6px; overflow:hidden; height:10px;">
                        <div style="background:#86b6ca; width:{pct*100:.1f}%; height:10px; border-radius:6px;"></div>
                    </div>
                    <div style="color:#FFFFFF; font-weight:700; font-size:26px; margin-top:8px;">{pct*100:.1f}%</div>

                </div>
                """, unsafe_allow_html=True)

            frame_index += 1

    cap.release()

    # ── Post-processing ────────────────────────────────────

    duration_sec = round(total_frames / fps, 2) if fps > 0 else 0
    duration_min = duration_sec / 60.0

    # ── Blink metrics ──
    blink_rate = round(blink_count / duration_min, 2) if duration_min > 0 else 0

    if blink_durations:
        avg_duration_ms = round(float(np.mean(blink_durations)), 2)
    else:
        avg_duration_ms = 0

    blink_intervals = [
        round(blink_timestamps[i] - blink_timestamps[i - 1], 3)
        for i in range(1, len(blink_timestamps))
    ]
    consistency = analyze_consistency(blink_intervals)
    blink_cv = consistency.get("cv")  # None if insufficient

    # Blink risk score
    blink_score = 0.0
    if blink_rate < NORMAL_BLINK_RATE_MIN:
        blink_score += (NORMAL_BLINK_RATE_MIN - blink_rate) / NORMAL_BLINK_RATE_MIN
    elif blink_rate > NORMAL_BLINK_RATE_MAX:
        blink_score += (blink_rate - NORMAL_BLINK_RATE_MAX) / NORMAL_BLINK_RATE_MAX

    if avg_duration_ms < NORMAL_BLINK_DURATION_MIN:
        blink_score += (NORMAL_BLINK_DURATION_MIN - avg_duration_ms) / NORMAL_BLINK_DURATION_MIN
    elif avg_duration_ms > NORMAL_BLINK_DURATION_MAX:
        blink_score += (avg_duration_ms - NORMAL_BLINK_DURATION_MAX) / NORMAL_BLINK_DURATION_MAX

    if blink_cv is not None:
        if blink_cv < 0.10:
            blink_score += (0.10 - blink_cv) / 0.10
        elif blink_cv > 0.60:
            blink_score += (blink_cv - 0.60) / 0.60

    blink_risk = round(clamp(blink_score / 3.0), 3)

    # ── Temporal metrics ──
    if face_width and len(landmarks_history) >= 2:
        temporal_stats = analyze_temporal(landmarks_history, fps, face_width)
        displacements   = temporal_stats["displacements"]
        jump_events     = temporal_stats["jump_events"]
        jump_score      = temporal_stats["jump_score"]
        freeze_score    = temporal_stats["freeze_score"]
        smooth_score    = temporal_stats["smooth_score"]
        jitter_flag     = temporal_stats["jitter_flag"]
        jump_threshold  = temporal_stats["jump_threshold_px"]
        freeze_threshold = temporal_stats["freeze_threshold_px"]
        jitter_cv       = temporal_stats["jitter_cv"]

        temporal_risk_score = (jump_score + freeze_score + smooth_score) / 3.0
        if jitter_flag != "NORMAL":
            temporal_risk_score += 0.2
        temporal_risk = round(clamp(temporal_risk_score), 3)
    else:
        displacements    = []
        jump_events      = 0
        jump_threshold   = 0
        freeze_threshold = FREEZE_THRESHOLD_PX
        jitter_cv        = 0
        jitter_flag      = "INSUFFICIENT DATA"
        temporal_risk    = 0.0
        temporal_stats   = {}

    # ── Symmetry metrics ──
    if len(landmarks_history) >= 2:
        symmetry_stats   = analyze_symmetry(landmarks_history)
        static_score     = symmetry_stats.get("static_score", 0)
        cv_score_sym     = symmetry_stats.get("cv_score", 0)
        motion_score     = symmetry_stats.get("motion_score", 0)
        regional_combined = symmetry_stats.get("regional_combined", 0)
        upper_score      = symmetry_stats.get("upper_score", 0)
        middle_score     = symmetry_stats.get("middle_score", 0)
        lower_score      = symmetry_stats.get("lower_score", 0)
        mean_static      = symmetry_stats.get("mean_static", 0)

        symmetry_risk = round(
            clamp((static_score + cv_score_sym + motion_score + regional_combined) / 4.0), 3
        )
    else:
        static_score     = 0
        cv_score_sym     = 0
        motion_score     = 0
        regional_combined = 0
        upper_score      = 0
        middle_score     = 0
        lower_score      = 0
        mean_static      = 0
        symmetry_risk    = 0.0
        symmetry_stats   = {}

    # ── Final risk score ──
    final_score = round(
        0.25 * blink_risk +
        0.40 * temporal_risk +
        0.35 * symmetry_risk,
        3
    )

    if final_score < 0.3:
        verdict = "Likely Real"
    elif final_score < 0.6:
        verdict = "Suspicious"
    else:
        verdict = "High Risk — Likely Deepfake"

    return {
        # Video info
        "fps":           round(fps, 2),
        "total_frames":  total_frames,
        "duration_sec":  duration_sec,

        # Blink
        "ear_values":       ear_values,
        "blink_frames":     blink_frames,
        "blink_count":      blink_count,
        "blink_rate":       blink_rate,
        "avg_duration_ms":  avg_duration_ms,
        "blink_cv":         blink_cv,
        "blink_risk":       blink_risk,
        "consistency_flag": consistency.get("consistency_flag", "INSUFFICIENT DATA"),

        # Temporal
        "displacements":     displacements,
        "jump_events":       jump_events,
        "jump_threshold":    jump_threshold,
        "freeze_threshold":  freeze_threshold,
        "jitter_cv":         jitter_cv,
        "jitter_flag":       jitter_flag,
        "temporal_risk":     temporal_risk,

        # Symmetry
        "static_asym_values": static_asym_values,
        "mean_static":        mean_static,
        "upper_score":        upper_score,
        "middle_score":       middle_score,
        "lower_score":        lower_score,
        "symmetry_risk":      symmetry_risk,

        # Summary
        "final_score":  final_score,
        "verdict":      verdict,
        "logs":         logs,
    }


# ─────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────

def create_gauge_chart(risk_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score * 100,
        delta={'reference': 50, 'suffix': " from baseline"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f3555"},
            'steps': [
                {'range': [0, 33],  'color': "#10B981"},
                {'range': [33, 66], 'color': "#F59E0B"},
                {'range': [66, 100],'color': "#EF4444"}
            ],
            'threshold': {
                'line': {'color': "#86b6ca", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        },
        number={'suffix': "%", 'font': {'size': 28}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(
        paper_bgcolor="#1f3555",
        plot_bgcolor="#1f3555",
        font={'color': "#B8C5D6", 'size': 12},
        margin={'l': 20, 'r': 20, 't': 20, 'b': 20},
        height=300
    )
    return fig


def create_ear_chart(ear_values, blink_frames, fps):
    frames = list(range(len(ear_values)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frames, y=ear_values,
        mode='lines',
        name='EAR Value',
        line=dict(color='#10B981', width=1.5),
        hovertemplate='<b>Frame:</b> %{x}<br><b>EAR:</b> %{y:.3f}<extra></extra>'
    ))
    for bf in blink_frames:
        fig.add_vline(x=bf, line_color="orange", line_width=0.8, opacity=0.5)
    fig.add_hline(y=EAR_THRESHOLD, line_dash="dash", line_color="#EF4444")
    fig.update_layout(
        title=dict(text="Eye Aspect Ratio (EAR) Over Time", font=dict(color="#86b6ca", size=14)),
        xaxis_title="Frames",
        yaxis_title="EAR Value",
        hovermode='x unified',
        paper_bgcolor="#1f3555",
        plot_bgcolor="#182c4d",
        font={'color': "#B8C5D6", 'size': 11},
        xaxis={'gridcolor': "#2a4a75"},
        yaxis={'gridcolor': "#2a4a75"},
        height=350,
        margin=dict(t=30, b=55, l=60, r=20),
        annotations=[dict(
            x=0, y=-0.26,
            xref="paper", yref="paper",
            text=(
                f'<span style="color:#EF4444;">— </span>'
                f'<span style="color:#B8C5D6;">Blink Threshold ({EAR_THRESHOLD})</span>'
                f'<br>'
                f'<span style="color:orange;">| </span>'
                f'<span style="color:#B8C5D6;">Blink Events</span>'
            ),
            showarrow=False,
            font=dict(size=10),
            xanchor="left",
            align="left"
        )]
    )
    return fig


def create_displacement_chart(displacements, jump_threshold, freeze_threshold):
    frames = list(range(len(displacements)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frames, y=displacements,
        mode='lines',
        name='Displacement (px)',
        line=dict(color='#F59E0B', width=1.5),
        hovertemplate='<b>Frame:</b> %{x}<br><b>Displacement:</b> %{y:.3f}px<extra></extra>'
    ))
    if jump_threshold:
        fig.add_hline(y=jump_threshold, line_dash="dot", line_color="#EF4444")
    fig.add_hline(y=freeze_threshold, line_dash="dot", line_color="#F59E0B")

    legend_parts = []
    if jump_threshold:
        legend_parts.append(
            f'<span style="color:#EF4444;">····</span>'
            f'<span style="color:#B8C5D6;"> Jump Threshold ({round(jump_threshold, 1)}px)</span>'
        )
    legend_parts.append(
        f'<span style="color:#F59E0B;">····</span>'
        f'<span style="color:#B8C5D6;"> Freeze Floor ({freeze_threshold}px)</span>'
    )

    fig.update_layout(
        title=dict(text="Landmark Displacement Velocity (per frame)", font=dict(color="#86b6ca", size=14)),
        xaxis_title="Frames",
        yaxis_title="Mean Displacement (px)",
        hovermode='x unified',
        paper_bgcolor="#1f3555",
        plot_bgcolor="#182c4d",
        font={'color': "#B8C5D6", 'size': 11},
        xaxis={'gridcolor': "#2a4a75"},
        yaxis={'gridcolor': "#2a4a75"},
        height=350,
        margin=dict(t=30, b=55, l=60, r=20),
        annotations=[dict(
            x=0, y=-0.26,
            xref="paper", yref="paper",
            text="<br>".join(legend_parts),
            showarrow=False,
            font=dict(size=10),
            xanchor="left",
            align="left"
        )]
    )
    return fig


def create_asymmetry_chart(static_asym_values):
    frames = list(range(len(static_asym_values)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frames, y=static_asym_values,
        mode='lines',
        name='Asymmetry Score',
        line=dict(color='#3B82F6', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)',
        hovertemplate='<b>Frame:</b> %{x}<br><b>Asymmetry:</b> %{y:.3f}<extra></extra>'
    ))
    fig.add_hrect(y0=0.05, y1=0.15, fillcolor="#10B981", opacity=0.15)
    fig.update_layout(
        title=dict(text="Static Facial Asymmetry Over Time", font=dict(color="#86b6ca", size=14)),
        xaxis_title="Frames",
        yaxis_title="Asymmetry Score",
        hovermode='x unified',
        paper_bgcolor="#1f3555",
        plot_bgcolor="#182c4d",
        font={'color': "#B8C5D6", 'size': 11},
        xaxis={'gridcolor': "#2a4a75"},
        yaxis={'gridcolor': "#2a4a75"},
        height=350,
        margin=dict(t=30, b=100, l=60, r=20),
        annotations=[dict(
            x=0, y=-0.46,
            xref="paper", yref="paper",
            text=(
                f'<span style="color:#10B981;">█ </span>'
                f'<span style="color:#B8C5D6;">Normal Range (0.05–0.15)</span>'
                f'<br>'
                f'<span style="color:#B8C5D6;">Below 0.05 = suspiciously symmetric</span>'
                f'<br>'
                f'<span style="color:#B8C5D6;">Above 0.15 = suspiciously asymmetric</span>'
            ),
            showarrow=False,
            font=dict(size=10),
            xanchor="left",
            align="left"
        )]
    )
    return fig


def create_symmetry_bar_chart(scores):
    regions = ['Upper (Brows)', 'Middle (Eyes)', 'Lower (Lips)']
    colors  = ['#10B981' if s < 0.1 else ('#F59E0B' if s < 0.2 else '#EF4444') for s in scores]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=regions, y=scores,
        marker=dict(color=colors),
        text=[f'{s:.3f}' for s in scores],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text="Regional Symmetry Suspicion Scores",font=dict(color="#86b6ca", size=14)),
        xaxis_title="Facial Region",
        yaxis_title="Suspicion Score (0–1)",
        paper_bgcolor="#1f3555",
        plot_bgcolor="#182c4d",
        font={'color': "#B8C5D6", 'size': 11},
        yaxis={'gridcolor': "#2a4a75", 'range': [0, 1]},
        showlegend=False,
        height=350
    )
    return fig


# ─────────────────────────────────────────────
#  FORENSIC REPORT
# ─────────────────────────────────────────────

def generate_forensic_report(r, timestamp):
    report = {
        "analysis_metadata": {
            "timestamp":  timestamp,
            "version":    "BD2FA v2.0",
            "model":      "Behavioral-Based Deepfake Detection"
        },
        "video_info": {
            "fps":          r["fps"],
            "total_frames": r["total_frames"],
            "duration_sec": r["duration_sec"],
        },
        "verdict": {
            "risk_score":     r["final_score"],
            "classification": r["verdict"]
        },
        "blink_analysis": {
            "total_blinks":       r["blink_count"],
            "blink_rate_per_min": r["blink_rate"],
            "avg_duration_ms":    r["avg_duration_ms"],
            "blink_cv":           r["blink_cv"],
            "consistency_flag":   r["consistency_flag"],
            "blink_risk":         r["blink_risk"],
        },
        "temporal_analysis": {
            "jump_events":   r["jump_events"],
            "jitter_cv":     r["jitter_cv"],
            "jitter_flag":   r["jitter_flag"],
            "temporal_risk": r["temporal_risk"],
        },
        "symmetry_analysis": {
            "mean_static_asymmetry": r["mean_static"],
            "upper_zone_score":      r["upper_score"],
            "middle_zone_score":     r["middle_score"],
            "lower_zone_score":      r["lower_score"],
            "symmetry_risk":         r["symmetry_risk"],
        },
        "analysis_logs": r["logs"]
    }
    return json.dumps(report, indent=2)


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────

def main():
    st.markdown(
        '<h1 style="color:#FFFFFF; font-size:70px; font-weight:800; margin-bottom:2px;">'
        'Behavioral-Based Deepfake Detection</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="color:#86b6ca; font-size:30px; font-weight:650; '
        'margin-top:-8px; margin-bottom:80px;">AI-Powered Forensic Analysis Dashboard (BD2FA)</p>',
        unsafe_allow_html=True
    )

    # Session state init
    for key, default in [
        ("analysis_complete", False),
        ("video_path", None),
        ("results", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # =====================================================================
    # UPLOAD PAGE
    # =====================================================================
    if not st.session_state.analysis_complete:
        st.markdown(
            '<h1 style="color:#86b6ca; margin-top:0; margin-bottom:12px;">Upload & Analysis 📤</h1>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Select video file (.mp4, .avi, .mov)",
            type=["mp4", "avi", "mov"],
            key="main_uploader"
        )

        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            st.session_state.video_path = video_path

            duration    = validate_video(video_path)
            frame_count = get_frame_count(video_path)
            fps         = get_video_fps(video_path)

            st.markdown(
                f'<div style="color:#86b6ca; padding:8px 0 4px 0; font-size:14px;">'
                f'Video duration: <b>{duration:.2f} seconds</b></div>',
                unsafe_allow_html=True
            )

            if duration < 30:
                st.warning("⚠️ **Low Confidence:** Video is under 30 seconds. Analysis may be less accurate.")

            if st.button("Start Analysis", type="primary", use_container_width=True, key="start_analysis"):
                st.markdown("---")
                st.markdown(
                    '<h1 style="color:#86b6ca; margin-bottom:8px;">Live Analysis Feed</h1>',
                    unsafe_allow_html=True
                )

                left_col, right_col = st.columns([1, 1], gap="medium")

                with left_col:
                    st.markdown(
                        '<p style="color:#B8C5D6; font-size:20px; margin-bottom:6px;">Live Video Feed :</p>',
                        unsafe_allow_html=True
                    )
                    feed_placeholder = st.empty()

                with right_col:
                    st.markdown(
                        '<p style="color:#B8C5D6; font-size:20px; margin-bottom:6px;">Video Analysis :</p>',
                        unsafe_allow_html=True
                    )
                    st.markdown(f"""
                    <div style="display:flex; gap:40px; margin-bottom:16px;">
                        <div>
                            <div style="color:#B8C5D6; font-size:15px;">Duration</div>
                            <div style="color:#FFFFFF; font-weight:700; font-size:18px;">{duration:.2f}s</div>
                        </div>
                        <div>
                            <div style="color:#B8C5D6; font-size:15px;">Total Frames</div>
                            <div style="color:#FFFFFF; font-weight:700; font-size:18px;">{frame_count}</div>
                        </div>
                        <div>
                            <div style="color:#B8C5D6; font-size:15px;">FPS</div>
                            <div style="color:#FFFFFF; font-weight:700; font-size:18px;">{fps:.1f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    progress_placeholder = st.empty()

                # ── Run real analysis ──
                results = run_analysis(video_path, feed_placeholder, progress_placeholder)
                st.session_state.results = results
                st.session_state.analysis_complete = True
                st.rerun()

    # =====================================================================
    # RESULTS DASHBOARD
    # =====================================================================
    else:
        r = st.session_state.results

        st.markdown('<h1 style="color: #86b6ca;">Analysis Results Dashboard</h1>', unsafe_allow_html=True)

        # Low-confidence banner
        if r["duration_sec"] < 30:
            st.warning("⚠️ **Low Confidence:** Video was under 30 seconds. Results may be less reliable.")

        st.divider()

        final_score  = r["final_score"]
        verdict      = r["verdict"]
        blink_risk   = r["blink_risk"]
        temporal_risk = r["temporal_risk"]
        symmetry_risk = r["symmetry_risk"]
        blink_cv     = r["blink_cv"]
        jump_events  = r["jump_events"]

        # ── SECTION 1: GAUGE + METRICS ──
        st.markdown("## 📋 Risk Assessment")
        gauge_col, metrics_col = st.columns([0.6, 0.4], gap="medium")

        with gauge_col:
            st.plotly_chart(create_gauge_chart(final_score), use_container_width=True)

        with metrics_col:
            st.markdown("### Key Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Blink Risk",
                    f"{blink_risk:.2f}",
                    delta=round(blink_risk - 0.30, 2),
                    delta_color="inverse",
                    help="Deviation from normal blink rate/duration/consistency"
                )
                st.metric(
                    "Jump Events",
                    f"{jump_events}",
                    help="Unnatural multi-frame facial geometry jumps"
                )
            with col2:
                st.metric(
                    "Symmetry Risk",
                    f"{symmetry_risk:.2f}",
                    delta=round(symmetry_risk - 0.30, 2),
                    delta_color="inverse",
                    help="Facial asymmetry deviations from natural range"
                )
                st.metric(
                    "Temporal Risk",
                    f"{temporal_risk:.2f}",
                    delta=round(temporal_risk - 0.30, 2),
                    delta_color="inverse",
                    help="Frame-to-frame motion anomalies"
                )

        st.divider()

        # ── SECTION 2: XAI SUMMARY ──
        st.markdown("## 🧠 Explainable AI Analysis")

        # Build dynamic explanation
        findings = []

        # Blink
        if r["blink_rate"] < NORMAL_BLINK_RATE_MIN:
            findings.append(f"abnormally low blink rate ({r['blink_rate']} blinks/min; normal: {NORMAL_BLINK_RATE_MIN}–{NORMAL_BLINK_RATE_MAX})")
        elif r["blink_rate"] > NORMAL_BLINK_RATE_MAX:
            findings.append(f"abnormally high blink rate ({r['blink_rate']} blinks/min; normal: {NORMAL_BLINK_RATE_MIN}–{NORMAL_BLINK_RATE_MAX})")

        if blink_cv is not None and blink_cv < 0.10:
            findings.append(f"robotically regular blink intervals (CV: {blink_cv:.3f}; normal: 0.10–0.60)")
        elif blink_cv is not None and blink_cv > 0.60:
            findings.append(f"erratically irregular blink intervals (CV: {blink_cv:.3f}; normal: 0.10–0.60)")

        if r["avg_duration_ms"] > 0 and r["avg_duration_ms"] < NORMAL_BLINK_DURATION_MIN:
            findings.append(f"abnormally short blink duration ({r['avg_duration_ms']}ms; normal: {NORMAL_BLINK_DURATION_MIN}–{NORMAL_BLINK_DURATION_MAX}ms)")
        elif r["avg_duration_ms"] > NORMAL_BLINK_DURATION_MAX:
            findings.append(f"abnormally long blink duration ({r['avg_duration_ms']}ms)")

        # Temporal
        if jump_events > 0:
            findings.append(f"{jump_events} geometric jump event(s) violating muscle biomechanics")
        if r["jitter_flag"] != "NORMAL":
            findings.append(f"motion jitter is {r['jitter_flag'].lower()} (CV: {r['jitter_cv']:.3f})")

        # Symmetry
        if symmetry_risk > 0.3:
            findings.append(
                f"facial asymmetry deviates from natural range "
                f"(mean static: {r['mean_static']:.3f}; normal: 0.05–0.15)"
            )

        if findings:
            finding_text = "; ".join(findings) + "."
            finding_text = finding_text[0].upper() + finding_text[1:]
        else:
            finding_text = "No significant anomalies detected across blink, temporal, or symmetry signals."

        st.info(f"**{verdict}** — {finding_text}")

        st.divider()

        # ── SECTION 3: CHARTS ──
        st.markdown("## 📊 Detailed Metrics")

        # EAR chart
        if r["ear_values"]:
            st.plotly_chart(
                create_ear_chart(r["ear_values"], r["blink_frames"], r["fps"]),
                use_container_width=True
            )
        else:
            st.info("No EAR data available (no face detected).")

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            if r["displacements"]:
                st.plotly_chart(
                    create_displacement_chart(
                        r["displacements"],
                        r["jump_threshold"],
                        r["freeze_threshold"]
                    ),
                    use_container_width=True
                )
            else:
                st.info("No displacement data available.")

        with col2:
            if r["static_asym_values"]:
                st.plotly_chart(
                    create_asymmetry_chart(r["static_asym_values"]),
                    use_container_width=True
                )
            else:
                st.info("No asymmetry data available.")

        regional_scores = [r["upper_score"], r["middle_score"], r["lower_score"]]
        st.plotly_chart(
            create_symmetry_bar_chart(regional_scores),
            use_container_width=True
        )

        st.divider()

        # ── SECTION 4: FORENSIC REPORT ──
        st.markdown("## 🗂️ Forensic Report Export")
        timestamp   = datetime.now().isoformat()
        json_report = generate_forensic_report(r, timestamp)

        st.download_button(
            label="📥 Download JSON Report",
            data=json_report,
            file_name=f"BD2FA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="json_download"
        )

        csv_data = (
            f"Analysis Timestamp,{timestamp}\n"
            f"Risk Score,{final_score:.3f}\n"
            f"Verdict,{verdict}\n"
            f"Blink Rate (blinks/min),{r['blink_rate']}\n"
            f"Avg Blink Duration (ms),{r['avg_duration_ms']}\n"
            f"Blink CV,{r['blink_cv'] if r['blink_cv'] is not None else 'N/A'}\n"
            f"Blink Risk,{blink_risk:.3f}\n"
            f"Jump Events,{jump_events}\n"
            f"Jitter CV,{r['jitter_cv']:.4f}\n"
            f"Temporal Risk,{temporal_risk:.3f}\n"
            f"Mean Static Asymmetry,{r['mean_static']:.4f}\n"
            f"Symmetry Risk,{symmetry_risk:.3f}\n"
        )
        st.download_button(
            label="📥 Download CSV Summary",
            data=csv_data,
            file_name=f"BD2FA_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="csv_download"
        )

        st.divider()

        # ── FOOTER ──
        _, col_center, _ = st.columns([1, 1, 1])
        with col_center:
            if st.button("Analyse Another Video", type="primary", use_container_width=True, key="back_btn"):
                st.session_state.analysis_complete = False
                st.session_state.results = None
                if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                    try:
                        os.unlink(st.session_state.video_path)
                    except Exception:
                        pass
                st.rerun()


if __name__ == "__main__":
    main()