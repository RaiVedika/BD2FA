import subprocess
import re


# ─────────────────────────────────────────────
# RUN MODULE AND CAPTURE OUTPUT
# ─────────────────────────────────────────────
def run_module(script, video_path):
    result = subprocess.run(
        ["python", script, video_path],
        capture_output=True,
        text=True
    )
    return result.stdout


# ─────────────────────────────────────────────
# SAFE EXTRACTION (NO CRASH)
# ─────────────────────────────────────────────
def safe_extract(pattern, text, name):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        print(f"[WARNING] Could not parse {name}")
        return 0.0


# ─────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────
def parse_blink(output):
    blink_rate = safe_extract(r"Blink Rate.*?([\d.]+)", output, "blink_rate")
    avg_duration = safe_extract(r"Avg Duration.*?([\d.]+)", output, "avg_duration")

    cv_match = re.search(r"CV.*?([\d.]+)", output)
    consistency = float(cv_match.group(1)) if cv_match else None

    return blink_rate, avg_duration, consistency


def parse_temporal(output):
    jump_score = safe_extract(r"Jump events.*?score.*?([\d.]+)", output, "jump_score")
    freeze_score = safe_extract(r"Freeze events.*?score.*?([\d.]+)", output, "freeze_score")
    smooth_score = safe_extract(r"Smooth violations.*?score.*?([\d.]+)", output, "smooth_score")

    jitter_match = re.search(r"Jitter status\s*:\s*(\w+)", output)
    jitter_flag = jitter_match.group(1) if jitter_match else "NORMAL"

    return jump_score, freeze_score, smooth_score, jitter_flag


def parse_symmetry(output):
    static_score = safe_extract(r"Static score.*?([\d.]+)", output, "static_score")
    cv_score     = safe_extract(r"CV score.*?([\d.]+)", output, "cv_score")
    motion_score = safe_extract(r"Motion score.*?([\d.]+)", output, "motion_score")
    regional     = safe_extract(r"Regional combined.*?([\d.]+)", output, "regional")

    return static_score, cv_score, motion_score, regional


# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────
def clamp(x):
    return max(0.0, min(x, 1.0))


# ─────────────────────────────────────────────
# RISK CALCULATIONS
# ─────────────────────────────────────────────
def compute_blink_risk(rate, duration, cv):
    score = 0

    # Blink rate
    if rate < 8:
        score += (8 - rate) / 8
    elif rate > 24:
        score += (rate - 24) / 24

    # Duration
    if duration < 100:
        score += (100 - duration) / 100
    elif duration > 400:
        score += (duration - 400) / 400

    # Consistency
    if cv is not None:
        if cv < 0.10:
            score += (0.10 - cv) / 0.10
        elif cv > 0.60:
            score += (cv - 0.60) / 0.60

    return clamp(score / 3.0)


def compute_temporal_risk(jump, freeze, smooth, jitter_flag):
    score = (jump + freeze + smooth) / 3.0

    if jitter_flag != "NORMAL":
        score += 0.2

    return clamp(score)


def compute_symmetry_risk(static, cv, motion, regional):
    return clamp((static + cv + motion + regional) / 4.0)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def analyze(video_path):

    print("\n===== RUNNING MODULES =====\n")

    blink_out = run_module("blink_analysis.py", video_path)
    temp_out  = run_module("temporal_analysis.py", video_path)
    sym_out   = run_module("symmetry_analysis.py", video_path)

    # Debug (optional — remove later)
    # print("\n--- BLINK OUTPUT ---\n", blink_out)
    # print("\n--- TEMPORAL OUTPUT ---\n", temp_out)
    # print("\n--- SYMMETRY OUTPUT ---\n", sym_out)

    print("\n===== PARSING OUTPUT =====\n")

    blink_rate, duration, cv = parse_blink(blink_out)
    jump, freeze, smooth, jitter_flag = parse_temporal(temp_out)
    static, cv_s, motion, regional = parse_symmetry(sym_out)

    print("\n===== COMPUTING RISK =====\n")

    blink_risk = compute_blink_risk(blink_rate, duration, cv)
    temporal_risk = compute_temporal_risk(jump, freeze, smooth, jitter_flag)
    symmetry_risk = compute_symmetry_risk(static, cv_s, motion, regional)

    final = round(
        0.25 * blink_risk +
        0.40 * temporal_risk +
        0.35 * symmetry_risk,
        3
    )

    print("\n===== FINAL RESULT =====")
    print(f"Blink Risk      : {blink_risk}")
    print(f"Temporal Risk   : {temporal_risk}")
    print(f"Symmetry Risk   : {symmetry_risk}")
    print(f"\n🔥 FINAL DEEPFAKE SCORE: {final}")

    if final < 0.3:
        print("→ LIKELY REAL")
    elif final < 0.6:
        print("→ SUSPICIOUS")
    else:
        print("→ HIGHLY LIKELY DEEPFAKE")

    return final


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main_risk.py <video_path>")
        exit()

    analyze(sys.argv[1])