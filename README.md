# BD2FA - Behavioral-Based Deepfake Detection using Facial Motion Analysis

A sophisticated Streamlit-based forensic analysis platform for detecting deepfake videos through behavioral and temporal analysis of facial motion patterns.

## Features

- **Real-time Video Analysis**: Process and analyze video files (MP4, AVI, MOV)
- **Interactive Risk Assessment**: Visual risk gauge with color-coded threat levels
- **Advanced Metrics**: 
  - Eye Aspect Ratio (EAR) analysis
  - Facial landmark displacement velocity
  - Facial asymmetry detection
  - Regional symmetry analysis
- **Temporal Anomaly Detection**: Frame-to-frame behavioral anomalies
- **Explainable AI**: Clear insights into detection methodology
- **Forensic Reports**: Download analysis results in JSON/CSV formats
- **Professional Dark Theme**: Eye-friendly interface designed for extended analysis sessions

## Technology Stack

- **Streamlit 1.57.0** - Web framework
- **MediaPipe 0.10.13** - Facial landmark detection
- **OpenCV 4.13.0.92** - Video processing
- **Plotly** - Interactive charting
- **NumPy** - Numerical computations

## Installation

### Requirements
- Python 3.11 (required for MediaPipe compatibility)
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd BD2FA
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The dashboard will be accessible at `http://localhost:8501`

### Workflow
1. Upload a video file (30+ seconds recommended)
2. Click "Start Analysis" to begin processing
3. View real-time analysis metrics and progress
4. Examine detailed forensic analysis results
5. Download forensic report in your preferred format

## Project Structure

```
BD2FA/
├── app.py                    # Main Streamlit application
├── blink_analysis.py        # Blink pattern analysis module
├── temporal_analysis.py      # Temporal anomaly detection
├── symmetry_analysis.py      # Facial symmetry analysis
├── main_risk.py             # Risk assessment aggregation
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Deployment on Streamlit Cloud

1. Push your code to a GitHub repository
2. Connect your GitHub account to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and select this repository
4. Streamlit will automatically install dependencies from `requirements.txt` and deploy

## Features In Development

- PDF forensic report export
- Real-time MediaPipe facial landmark visualization
- Database persistence for analysis history
- Multi-video batch processing

## License

This project is part of the BD2FA research initiative.

## Contact

For questions or support, please reach out to the project team.
