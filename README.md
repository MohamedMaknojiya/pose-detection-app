Pose Detection App

A PyQt5 and MediaPipe application for automated, real-time monitoring of attention and pose in classes or meetings.

1. Business Problems & Solutions

Problems:

Manual tracking of participant attention is error-prone and resource-intensive.

Lack of objective metrics for engagement over time.

Difficult to generate historical reports for training or productivity analysis.

Solutions:

Real-time pose and face landmark analysis to detect focus and engagement.

Automated logging of attention levels and session data to a database.

Visual graphs and historical session reports for data-driven insights.

2. Solution Details

2.1 Tech Environment

Language: Python 3.8+

GUI: PyQt5 (v5.15.x)

Computer Vision: OpenCV (v4.x), MediaPipe (v0.8.x)

ML: scikit-learn (v1.x) for model inference

Visualization: Matplotlib (v3.x)

Database: SQLite3 (builtin)

Version Control: Git & GitHub

2.2 Model Details

Model File: EmployeeDetection.pkl

Type: scikit-learn classifier (e.g. RandomForest)

Input: Flattened 3D pose and face landmark coordinates + visibility scores

Output: Body language class labels and probabilities (e.g. looking_straight)

2.3 Database Details

Filename: classes.db

Schema Versions: v1

classes table: scheduled class sessions

sessions table: session start, end, and duration

session_data table: timestamped pose and attention level logs

2.4 URLs

GitHub Repository: https://github.com/<your_username>/pose-detection-app

Documentation Folder: /docs

3. Problems Faced

Data Duplication: Duplicate session_data entries when landmark detection flickers.

Frame Skips: Inconsistent frame capture rates causing uneven time series data.

Cross-Platform Packaging: Bundling Qt and CV binaries for Windows/macOS.

4. Data Preprocessing

Extract landmark arrays ([x, y, z, visibility]) for pose and face from MediaPipe.

Flatten and concatenate into a single feature vector.

Handle missing landmarks by zero-padding.

Create a pandas.DataFrame for batch inference.

5. Deployment & Commands

# Clone repository
git clone https://github.com/<your_username>/pose-detection-app.git
cd pose-detection-app

# Optional: rename main script
tools/rename main.py to app.py (or leave as-is)

# Create virtual environment
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize database (if not exists)
python - << 'EOF'
from app import initialize_database
initialize_database()
EOF

# Run the application
python app.py

6. Upload Business Documents

Place any PDFs, Word docs, or spreadsheets under the /docs directory.

Commit and push to keep business reference files versioned alongside the code.
