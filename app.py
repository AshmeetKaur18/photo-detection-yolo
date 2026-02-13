import streamlit as st
import pandas as pd
from PIL import Image
import io
import zipfile

# Lazy load ultralytics only when needed
def load_yolo_model(model_path):
    from ultralytics import YOLO
    return YOLO(model_path)


# ============================================
# CONFIGURATION & SETTINGS
# ============================================
st.set_page_config(page_title="YOLO Object Detection Project", layout="wide")

st.title("📸 YOLO Object Detection Project")
st.markdown("Upload images to detect objects using YOLOv8 models. Compare performance and view detailed analytics.")

# Sidebar for Settings
st.sidebar.header("⚙️ Settings")

# Model Selection
available_models = {
    "YOLOv8n (Nano)": "yolov8n.pt",
    "YOLOv8l (Large)": "yolov8l.pt",
    "YOLOv8x (Extra Large)": "yolov8x.pt"
}
selected_models_names = st.sidebar.multiselect(
    "Select Models to Run",
    options=list(available_models.keys()),
    default=["YOLOv8n (Nano)"]
)

# Detection Control Rules
st.sidebar.subheader("Detection Rules")
ALLOWED_CLASSES = [
    "chair",
    "couch",
    "person",
    "vase",
    "potted plant"
]
selected_classes = st.sidebar.multiselect(
    "Allowed Classes",
    options=ALLOWED_CLASSES,
    default=ALLOWED_CLASSES
)

# Confidence Thresholds
st.sidebar.subheader("Confidence Thresholds")
class_conf_thresholds = {}
default_conf = st.sidebar.slider("Default Confidence Threshold", 0.0, 1.0, 0.40, 0.05)

with st.sidebar.expander("Advanced Class Thresholds"):
    for cls in ALLOWED_CLASSES:
        # Default values from the user's script
        default_val = 0.40
        if cls == "chair" or cls == "couch": default_val = 0.35
        elif cls == "person": default_val = 0.40
        elif cls == "vase": default_val = 0.60
        elif cls == "potted plant": default_val = 0.55
        
        class_conf_thresholds[cls] = st.slider(f"{cls.capitalize()} Threshold", 0.0, 1.0, default_val, 0.05)

# ============================================
# MAIN APPLICATION LOGIC
# ============================================

# 1. Upload Images
uploaded_files = st.file_uploader("Upload Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    st.write(f"**Total images uploaded:** {len(uploaded_files)}")
    
    if st.button("🚀 Run Detection"):
        if not selected_models_names:
            st.error("Please select at least one model from the sidebar.")
        else:
            records = []
            uncertain_logs = []
            cropped_objects = [] # Store cropped images
            annotated_images = {} # Store annotated images for ZIP download
            
            # Progress bar
            progress_bar = st.progress(0)
            total_steps = len(selected_models_names) * len(uploaded_files)
            step_counter = 0

            # Process for each selected model
            for model_display_name in selected_models_names:
                model_name = available_models[model_display_name]
                st.write(f"---")
                st.subheader(f"Running Model: {model_display_name}")
                
                try:
                    with st.spinner(f"Loading {model_display_name}..."):
                        model = load_yolo_model(model_name)
                except Exception as e:
                    st.error(f"Error loading model {model_name}: {e}")
                    continue

                # Grid for images
                cols = st.columns(3)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Convert to PIL Image
                    image = Image.open(uploaded_file)
                    
                    # Run inference
                    # We pass the PIL image directly to YOLO
                    results = model(image, conf=0.25, iou=0.35)
                    detections = results[0].boxes

                    # Prepare to visualize
                    res_plotted = results[0].plot() # BGR numpy array
                    res_image = Image.fromarray(res_plotted[..., ::-1]) # RGB
                    
                    # Save annotated image for ZIP
                    img_byte_arr = io.BytesIO()
                    res_image.save(img_byte_arr, format='JPEG')
                    annotated_images[f"{model_display_name}_{uploaded_file.name}"] = img_byte_arr.getvalue()

                    # Show image in grid
                    with cols[idx % 3]:
                        st.image(res_image, caption=f"{uploaded_file.name} ({model_display_name})", use_container_width=True)

                    if detections is None:
                        step_counter += 1
                        progress_bar.progress(min(step_counter / total_steps, 1.0))
                        continue

                    # Process detections
                    for box in detections:
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id]
                        confidence = float(box.conf[0])

                        # ❌ Ignore irrelevant classes (User Rule)
                        if class_name not in selected_classes:
                            continue

                        threshold = class_conf_thresholds.get(class_name, default_conf)

                        # ⚠️ Log uncertain detections (User Rule)
                        if confidence < threshold:
                            uncertain_logs.append({
                                "image": uploaded_file.name,
                                "model": model_display_name,
                                "class": class_name,
                                "confidence": confidence
                            })
                            continue

                        # ✅ Accept reliable detections (User Rule)
                        records.append({
                            "image": uploaded_file.name,
                            "model": model_display_name,
                            "class": class_name,
                            "confidence": confidence
                        })

                        # Extract cropped object
                        try:
                            # Box coordinates: x1, y1, x2, y2
                            box_coords = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = map(int, box_coords)
                            crop = image.crop((x1, y1, x2, y2))
                            
                            cropped_objects.append({
                                "image_name": uploaded_file.name,
                                "class_name": class_name,
                                "confidence": confidence,
                                "crop": crop,
                                "model": model_display_name
                            })
                        except Exception as e:
                            st.warning(f"Could not crop object: {e}")
                    
                    step_counter += 1
                    progress_bar.progress(min(step_counter / total_steps, 1.0))

            progress_bar.empty()
            
            # ============================================
            # RESULTS DISPLAY
            # ============================================
            
            df = pd.DataFrame(records)
            uncertain_df = pd.DataFrame(uncertain_logs)

            st.divider()
            st.header("📊 Detection Results")

            if df.empty:
                st.warning("⚠️ No reliable detections found matching your criteria.")
            else:
                # Summary Tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Reliable Detections", "Uncertain Detections", "Model Comparison", "🖼️ Cropped Objects", "📥 Downloads"])

                with tab1:
                    st.subheader("Reliable Detections Summary")
                    st.dataframe(df, use_container_width=True)
                    
                    # Statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Count by Class**")
                        st.bar_chart(df["class"].value_counts())
                    with col2:
                        st.write("**Average Confidence by Class**")
                        st.dataframe(df.groupby("class")["confidence"].mean().reset_index(), hide_index=True)

                with tab2:
                    st.subheader("Uncertain / Possible Errors")
                    if uncertain_df.empty:
                        st.success("No uncertain detections logged!")
                    else:
                        st.warning(f"Found {len(uncertain_df)} uncertain detections.")
                        st.dataframe(uncertain_df, use_container_width=True)
                        st.write("**Uncertainty by Class**")
                        st.bar_chart(uncertain_df["class"].value_counts())

                with tab3:
                    st.subheader("Final Model Comparison")
                    comparison = df.groupby("model").agg(
                        images_processed=("image", "nunique"),
                        reliable_detections=("class", "count"),
                        avg_confidence=("confidence", "mean")
                    ).reset_index()
                    
                    st.dataframe(comparison, use_container_width=True)

                    if not comparison.empty:
                        best_model = comparison.loc[comparison["avg_confidence"].idxmax()]["model"]
                        st.success(f"✅ Best performing model based on confidence: **{best_model}**")
                        
                        st.info("ℹ️ Note: Higher confidence doesn't always mean better accuracy if the model misses objects (False Negatives).")

                with tab4:
                    st.subheader("🖼️ Cropped Objects Gallery")
                    if not cropped_objects:
                        st.info("No objects cropped.")
                    else:
                        # Group by class for better organization
                        unique_classes = sorted(list(set(obj["class_name"] for obj in cropped_objects)))
                        selected_crop_class = st.selectbox("Filter by Class", ["All"] + unique_classes)

                        filtered_crops = cropped_objects if selected_crop_class == "All" else [obj for obj in cropped_objects if obj["class_name"] == selected_crop_class]
                        
                        crop_cols = st.columns(5) # Grid layout
                        for i, obj in enumerate(filtered_crops):
                            with crop_cols[i % 5]:
                                st.image(obj["crop"], caption=f"{obj['class_name']} ({obj['confidence']:.2f})", use_container_width=True)

                with tab5:
                    st.subheader("📥 Download Results")
                    
                    col_d1, col_d2 = st.columns(2)
                    
                    with col_d1:
                        st.write("### 📄 Data Report")
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Detection CSV",
                            data=csv,
                            file_name='detection_results.csv',
                            mime='text/csv',
                        )

                    with col_d2:
                        st.write("### 📸 Annotated Images")
                        # Create ZIP file in memory
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w") as zf:
                            for name, data in annotated_images.items():
                                zf.writestr(name, data)
                        
                        st.download_button(
                            label="Download All Images (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name="annotated_images.zip",
                            mime="application/zip"
                        )

else:
    st.info("👈 Please upload images from the sidebar or main area to start.")
