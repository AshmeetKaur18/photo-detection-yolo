# 📸 YOLO Object Detection Project

This is a "project-ready" implementation of an object detection application using YOLOv8 and Streamlit.

## 🚀 Features

- **User Interface**: A clean web-based interface built with Streamlit.
- **Multiple Models**: Compare YOLOv8 Nano, Large, and X-Large models.
- **Custom Rules**: Filter by specific classes (Chair, Couch, Person, Vase, Potted Plant).
- **Confidence Control**: Adjust confidence thresholds per class.
- **Detailed Analytics**: View reliable detections, uncertain detections, and model comparison charts.

## 🛠️ Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ How to Run

1.  Run the Streamlit app:
    ```bash
    python -m streamlit run app.py
    ```

    *Note: If `streamlit run app.py` doesn't work, using `python -m streamlit ...` ensures it uses the installed Python module.*

2.  The application will open in your default web browser (usually at `http://localhost:8501`).

3.  **Upload Images**: Use the sidebar or main upload area to select images.
4.  **Select Models**: Choose which YOLO models to run from the sidebar.
5.  **Run Detection**: Click the "Run Detection" button to process images.

## 📂 Project Structure

- `app.py`: The main application code containing the UI and logic.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This documentation file.
