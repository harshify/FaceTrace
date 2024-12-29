# FaceTrace - Face Recognition Attendance System

This repository contains a Python-based Face Recognition Attendance System that utilizes a K-Nearest Neighbors (KNN) classifier for efficient and accurate face recognition. The system is designed to automate attendance marking by capturing, training, and recognizing faces in real-time, while also logging attendance data.

---

## Features

- **Face Registration**: Capture and store new user facial data for training.
- **Face Recognition**: Identify registered users in real-time and mark attendance.
- **Streamlit Dashboard**: View daily attendance records in an interactive interface.
- **CSV Logs**: Automatically generate and save timestamped attendance records in CSV format.

---

## System Overview

### 1. Face Data Registration (`add_faces.py`)
- Captures facial images using the webcam.
- Processes and stores the facial data locally for training.
- Dynamically updates the dataset with new user entries.

### 2. Real-Time Recognition and Attendance Logging (`test.py`)
- Leverages a pre-trained KNN model to recognize faces in real-time.
- Logs attendance into a timestamped CSV file.
- Displays the live webcam feed with user names overlayed.

### 3. Attendance Viewer (`app.py`)
- A lightweight dashboard created using Streamlit.
- Allows users to view the current day's attendance dynamically.

---

## Prerequisites

Ensure the following are installed before proceeding:
- **Python**: Version 3.7 or above
- **Libraries**:
  - OpenCV
  - NumPy
  - scikit-learn
  - pandas
  - Streamlit
- **Webcam**: Required for capturing and recognizing faces.
- `haarcascade_frontalface_default.xml`: Included in OpenCV's data for face detection.

---

## Setup Instructions

### 1. Clone the Repository:
```bash
git clone https://github.com/Arpit0417/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System
```

### 2. Install Required Dependencies:
```bash
pip install -r requirements.txt
```

### 3. Prepare the Directory Structure:
- Create a folder named `data/` to store facial data and user names.
- Create a folder named `Attendance/` to save attendance logs.

### 4. Run the Scripts:
- **Add New Users**: `python add_faces.py`
- **Start Attendance Recognition**: `python test.py`
- **View Attendance Records**: `streamlit run app.py`

---

## Usage Instructions

### Adding New Users:
1. Run `add_faces.py` and enter the user's name when prompted.
2. The script will capture and store 100 facial images for training.

### Taking Attendance:
1. Run `test.py` to start the real-time recognition process.
2. Press `o` to log attendance for the detected user.
3. Press `c` to stop the recognition process.

### Viewing Attendance:
1. Run `app.py` to launch the Streamlit dashboard.
2. The dashboard will display the day's attendance records.

---

## Directory Structure

```
├── data/
│   ├── names.pkl                   # Stores user names
│   ├── faces_data.pkl              # Stores facial data
│   └── haarcascade_frontalface_default.xml
├── Attendance/
│   ├── Attendance_<date>.csv       # Daily attendance records
├── add_faces.py                    # Script for capturing user faces
├── test.py                         # Script for real-time recognition
├── app.py                          # Streamlit dashboard
└── requirements.txt                # Python package dependencies
```

---

## Future Enhancements

- Add support for detecting face masks during recognition.
- Implement multi-role support (e.g., staff and students).
- Migrate attendance records to a database for scalability.
- Optimize the KNN model for enhanced performance.

---

## Acknowledgements

- **OpenCV**: For face detection.
- **scikit-learn**: For KNN-based classification.
- **Streamlit**: For the interactive attendance dashboard.

---
