# 📸 YOLOv8 Multi-Model Object Detection Dashboard

An interactive, high-performance web application built with **Streamlit** and **YOLOv8** to detect, compare, and extract objects from images with real-time precision.

---

## 🚀 Features

### 🔍 Advanced Detection
- **Multi-Model Comparison**: Run **YOLOv8 Nano, Large, and X-Large** simultaneously to evaluate speed vs. accuracy.
- **Dynamic Rule Engine**: 
    - Filter by specific classes: *Chair, Couch, Person, Vase, Potted Plant*.
    - Set **individual confidence thresholds** per class via an interactive sidebar.
- **Lazy Loading**: Optimized performance with heavy AI dependencies loaded only when needed.

### 🖼️ Deep Extraction
- **Object Gallery**: Automatically extracts every detected object into a dedicated crop gallery.
- **Class Filtering**: View specific categories of detected objects with a single click.

### 📊 Data & Analytics
- **Live Statistics**: Real-time charts showing detection counts and average confidence levels.
- **Uncertainty Logs**: Separate tracking for low-confidence detections to ensure transparency.
- **Model Comparison**: Automated table ranking models based on their performance on your dataset.

### 📥 One-Click Exports
- **CSV Reports**: Export all detection data (class, confidence, model used) for further analysis in Excel.
- **ZIP Archives**: Download all processed images with bounding boxes in one compressed file.

---

## 🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AshmeetKaur18/photo-detection-yolo.git
   cd photo-detection-yolo
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

### Windows (Easy Way)
Simply double-click the **`run_app.bat`** file in the project folder.

### Terminal
Run the following command:
```bash
python -m streamlit run app.py
```
The app will open automatically at `http://localhost:8501`.

---

## 📂 Project Structure

- **`app.py`**: The core application logic and Streamlit UI.
- **`requirements.txt`**: Python library dependencies.
- **`run_app.bat`**: One-click execution script for Windows.
- **`.gitignore`**: Excludes large model weights and local configs from Git.
- **`.streamlit/`**: UI configuration settings.

---

## 🤝 Contributing
Feel free to fork this project, open issues, or submit pull requests to improve the detection logic or UI!

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
