# Pose Detection App

A PyQt5 & MediaPipe application for automated, real‑time monitoring of attention and pose in classes or meetings.

---

## 1. Business Problems & Solutions

**Problems:**
- Manual tracking of participant attention is time‑consuming and prone to human error.
- Lack of quantitative metrics for engagement over time.
- Difficulty generating historical reports for training or productivity analysis.

**Solutions:**
- Real‑time pose and face landmark detection to assess focus (using MediaPipe & OpenCV).
- Automated logging of attention levels and session data to a SQLite database.
- Interactive GUI (PyQt5) with live graphs and historical data views for insights.

---

## 2. Solution Details

### 2.1 Tech Environment & Versions
- **Programming Language:** Python 3.8+
- **GUI Framework:** PyQt5 5.15.11
- **Computer Vision:** OpenCV 4.11.0.86 & OpenCV‑contrib 4.11.0.86
- **Landmark Detection:** MediaPipe 0.10.21
- **Machine Learning:** scikit‑learn 1.6.1
- **Data Handling:** pandas 2.2.3, numpy 1.24.1
- **Visualization:** Matplotlib 3.6.1
- **Database:** SQLite3 (builtin, version depends on OS)
- **Other Key Dependencies:**
  - torch 1.12.1, torchvision 0.13.1, torchaudio 0.12.1
  - jupyterlab 3.4.8 (for analysis & prototyping)
  - Flask 3.1.0 (optional web dashboard)

> See full dependency list in `requirements.txt`.

### 2.2 Model Details
- **Model File:** `EmployeeDetection.pkl`  
- **Type:** scikit‑learn classifier (e.g. RandomForest)  
- **Input:** Flattened `[x,y,z,visibility]` arrays from pose & face landmarks  
- **Output:** Body language class label + probability distribution (e.g. `looking_straight`)

### 2.3 Database Details
- **Filename:** `classes.db`  
- **Schema (v1):**
  - **classes**: scheduled session records (id, date, time, topic, module)
  - **sessions**: session logs (session_id, class_id, start_time, end_time, duration)
  - **session_data**: timestamped pose & attention metrics (data_id, session_id, timestamp, pose, attention_level)

### 2.4 URLs
- **GitHub Repo:** `https://github.com/<your_username>/pose-detection-app`  
- **Documentation Folder:** `/docs`

---

## 3. Problems Faced
- **Data Duplication:** Duplicate rows in `session_data` when landmarks sporadically drop in/out.
- **Frame Skips:** Varying CPU load causing uneven frame capture intervals.
- **Packaging Qt:** Bundling PyQt5 dependencies for Windows/macOS distributions.

---

## 4. Data Preprocessing
1. Extract pose and face landmark lists from MediaPipe results.  
2. Convert each landmark to `[x, y, z, visibility]`.  
3. Flatten and concatenate pose + face arrays into a single feature vector.  
4. Zero‑pad missing landmarks when detection fails.  
5. Build a `pandas.DataFrame` for batch prediction by the ML model.

---

## 5. Deployment & Commands
```bash
# 1. Clone repository
git clone https://github.com/<your_username>/pose-detection-app.git
cd pose-detection-app

# 2. (Optional) rename entry script
mv main.py app.py

# 3. Create & activate virtual environment
python3 -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Initialize database (if not exists)
python -c "from app import initialize_database; initialize_database()"

# 6. Run the app
python app.py
```

---
