import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image, ImageFont, ImageDraw
import time
from pathlib import Path
import uuid
from datetime import datetime
from collections import defaultdict

class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobStore:
    def __init__(self):
        self.jobs = defaultdict(dict)
        
    def set_job_status(self, job_id, status, **kwargs):
        self.jobs[job_id].update({"status": status, **kwargs})
        
    def get_job_status(self, job_id):
        return self.jobs.get(job_id, {})

FONT_FAMILIES = {
    "Merriweather Regular": "Merriweather-Regular.ttf",
    "Merriweather Bold": "Merriweather-Bold.ttf",
    "Merriweather Light": "Merriweather-Light.ttf",
    "Merriweather Black": "Merriweather-Black.ttf",
    "Merriweather Italic": "Merriweather-Italic.ttf",
    "Merriweather Bold Italic": "Merriweather-BoldItalic.ttf"
}

TARGET_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'TV',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair dryer',
    79: 'toothbrush'}

def get_pil_font(font_family, font_size):
    try:
        font_path = Path(f"fonts/{FONT_FAMILIES[font_family]}")
        return ImageFont.truetype(str(font_path), font_size)
    except Exception:
        return ImageFont.load_default()

@st.cache_resource
def load_model():
    model = YOLO('yolov8s-seg.pt')
    model.conf = 0.5
    model.iou = 0.7
    return model

def add_text_with_pil(image, text, font_family, font_size, font_color, position, 
                     opacity=1.0, rotation=0, font_weight="normal", x_pos=0, y_pos=0):
    image = image.astype(np.uint8)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_rgba = image_pil.convert('RGBA')
    
    overlay = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    font = get_pil_font(font_family, font_size)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = x_pos if x_pos else (image_pil.width - text_width) // 2
    y = y_pos if y_pos else (image_pil.height - text_height) // 2
    
    if isinstance(font_color, str):
        font_color = tuple(int(font_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    font_color_with_opacity = (*font_color, int(255 * opacity))
    
    if rotation != 0:
        temp_size = int(np.sqrt(text_width**2 + text_height**2)) + 20
        temp_img = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        
        temp_x = (temp_size - text_width) // 2
        temp_y = (temp_size - text_height) // 2
        temp_draw.text((temp_x, temp_y), text, font=font, fill=font_color_with_opacity)
        
        rotated_text = temp_img.rotate(rotation, expand=True, resample=Image.BICUBIC)
        paste_x = x - (rotated_text.width - text_width) // 2
        paste_y = y - (rotated_text.height - text_height) // 2
        overlay.paste(rotated_text, (paste_x, paste_y), rotated_text)
    else:
        draw.text((x, y), text, font=font, fill=font_color_with_opacity)
    
    result = Image.alpha_composite(image_rgba, overlay)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGBA2BGR)

def update_preview(frame, text_sets, detected_objects=None):
    if frame is None:
        return None
    
    # Create the text overlay on a separate frame
    frame_with_text = frame.copy()
    for text_set in text_sets:
        if text_set.get("text"):
            frame_with_text = add_text_with_pil(
                frame_with_text,
                text_set["text"],
                text_set["font_family"],
                text_set["font_size"],
                text_set["font_color"],
                "center",
                text_set["opacity"],
                text_set["rotation"],
                text_set["font_weight"],
                text_set["x_pos"],
                text_set["y_pos"]
            )
    
    if detected_objects:
        # Create combined mask from all detected objects
        mask_combined = np.zeros((frame.shape[0], frame.shape[1]))
        for mask in detected_objects:
            mask_combined = np.maximum(mask_combined, mask)
        
        # Create a 3-channel mask for proper masking
        mask_3channel = np.stack([mask_combined] * 3, axis=-1)
        
        # Convert to uint8 and ensure proper data type
        mask_3channel = (mask_3channel * 255).astype(np.uint8)
        frame = frame.astype(np.uint8)
        frame_with_text = frame_with_text.astype(np.uint8)
        
        # Apply inverse masks to both frames
        frame = cv2.multiply(frame, mask_3channel, scale=1/255)
        frame_with_text = cv2.multiply(frame_with_text, 255 - mask_3channel, scale=1/255)
        
        # Combine the frames
        preview_frame = cv2.add(frame, frame_with_text)
    else:
        preview_frame = frame_with_text
    
    # Ensure output is uint8
    preview_frame = preview_frame.astype(np.uint8)
    return preview_frame

def get_first_frame(video_file):
    if not video_file:
        return None
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.getvalue())
        video_path = tmp_file.name
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    os.remove(video_path)
    
    return frame.astype(np.uint8) if ret else None

def process_video(video_file, text_sets, selected_classes, model, job_store, job_id,
                 confidence_threshold=0.5, iou_threshold=0.7):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name
        
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = f'output_video_{job_id}.mp4'
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
        
        model.conf = confidence_threshold
        model.iou = iou_threshold
        
        selected_class_ids = [k for k, v in TARGET_CLASSES.items() if v in selected_classes]
        progress_bar = st.progress(0)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame)
            mask_combined = np.zeros((height, width))
            
            for result in results:
                for box_idx, cls in enumerate(result.boxes.cls):
                    if int(cls) in selected_class_ids:
                        try:
                            mask = result.masks.data[box_idx].numpy()
                            mask = cv2.resize(mask, (width, height))
                            mask_combined = np.maximum(mask_combined, np.where(mask >= 0.3, 1.0, 0))
                        except (IndexError, AttributeError):
                            continue
            
            frame_with_text = frame.copy()
            for text_set in text_sets:
                frame_with_text = add_text_with_pil(
                    frame_with_text,
                    text_set["text"],
                    text_set["font_family"],
                    text_set["font_size"],
                    text_set["font_color"],
                    "center",
                    text_set["opacity"],
                    text_set["rotation"],
                    text_set["font_weight"],
                    text_set["x_pos"],
                    text_set["y_pos"]
                )
            
            # Apply masks
            frame[mask_combined == 0] = 0
            frame_with_text[mask_combined == 1] = 0
            final_frame = frame + frame_with_text
            
            out.write(final_frame)
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
        
        cap.release()
        out.release()
        os.remove(video_path)
        
        job_store.set_job_status(job_id, JobStatus.COMPLETED, result_path=output_path)
        return True
        
    except Exception as e:
        job_store.set_job_status(job_id, JobStatus.FAILED, error=str(e))
        return False

def main():
    st.set_page_config(page_title="Video Text Overlay App", page_icon="🎥", layout="wide")
    
    if 'job_store' not in st.session_state:
        st.session_state.job_store = JobStore()
    
    model = load_model()
    
    # Clean, modern header
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: 2rem;'>Video Text Overlay</h1>
        <p style='text-align: center; color: #666; margin-bottom: 3rem;'>Add professional text overlays to your videos with smart object detection</p>
    """, unsafe_allow_html=True)
    
    # Main content with better spacing
    upload_col, preview_col = st.columns([1, 2])
    
    with upload_col:
        st.markdown("### Upload Video")
        video_file = st.file_uploader("", type=["mp4", "mov", "avi"])
        
        if video_file:
            st.video(video_file)
            
        # Detection Settings in a clean card
        st.markdown("### Detection Settings")
        selected_classes = st.multiselect(
            "Objects to Detect",
            options=list(TARGET_CLASSES.values()),
            default=['person']
        )
        
        col1, col2 = st.columns(2)
        with col1:
            confidence_threshold = st.slider("Confidence", 0.0, 1.0, 0.5, 0.05)
        with col2:
            iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.7, 0.05)
    
    # Preview column
    with preview_col:
        if video_file and 'first_frame' not in st.session_state:
            first_frame = get_first_frame(video_file)
            if first_frame is not None:
                st.session_state['first_frame'] = first_frame
                st.session_state['frame_height'] = first_frame.shape[0]
                st.session_state['frame_width'] = first_frame.shape[1]
        
        # Modern tab design for preview with detection first
        if 'first_frame' in st.session_state:
            preview_tabs = st.tabs(["With Detection", "Preview"])
            
            # Initialize text sets in session state
            if 'text_sets' not in st.session_state:
                st.session_state.text_sets = [{}]
            
            # Handle layer removal
            layers_to_remove = []
            
            # Text settings in a cleaner layout
            text_sets = []
            for i, _ in enumerate(st.session_state.text_sets):
                with st.container():
                    # Layer header with remove button
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"#### Text Layer {i+1}")
                    with col2:
                        if i > 0:  # Only show remove button for layers after the first
                            if st.button("🗑️ Remove", key=f"remove_{i}"):
                                layers_to_remove.append(i)
                    
                    # Text input and basic settings
                    text_input = st.text_area("Text Content", key=f"text_{i}", 
                                            help="Enter the text you want to display")
                    
                    # Three columns for better organization
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        font_family = st.selectbox("Font", 
                                                 options=list(FONT_FAMILIES.keys()),
                                                 key=f"font_{i}")
                        font_size = st.slider("Size", 10, 500, 150, key=f"size_{i}")
                    
                    with c2:
                        max_x = st.session_state.get('frame_width', 1920)
                        max_y = st.session_state.get('frame_height', 1080)
                        x_pos = st.slider("X Position", 0, max_x, max_x//2, key=f"x_{i}")
                        y_pos = st.slider("Y Position", 0, max_y, max_y//2, key=f"y_{i}")
                    
                    with c3:
                        opacity = st.slider("Opacity", 0.0, 1.0, 1.0, 0.1, key=f"opacity_{i}")
                        rotation = st.slider("Rotation", -180, 180, 0, key=f"rotation_{i}")
                    
                    # Color and weight in a row
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        font_color = st.color_picker("Color", "#FFFFFF", key=f"color_{i}")
                    with col2:
                        font_weight = st.select_slider("Weight", 
                                                     options=["light", "normal", "bold", "black"],
                                                     value="normal",
                                                     key=f"weight_{i}")
                    
                    if text_input:
                        text_set = {
                            "text": text_input,
                            "font_family": font_family,
                            "font_size": font_size,
                            "font_color": font_color,
                            "x_pos": x_pos,
                            "y_pos": y_pos,
                            "opacity": opacity,
                            "rotation": rotation,
                            "font_weight": font_weight
                        }
                        text_sets.append(text_set)
                    
                    st.markdown("<hr style='margin: 2rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
            
            # Remove layers marked for removal
            for i in reversed(layers_to_remove):
                st.session_state.text_sets.pop(i)
                st.rerun()
            
            # Add text layer button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Add Text Layer", use_container_width=True):
                    st.session_state.text_sets.append({})
            
            # Show detection preview first
            with preview_tabs[0]:
                selected_class_ids = [k for k, v in TARGET_CLASSES.items() if v in selected_classes]
                results = model(st.session_state['first_frame'])
                detected_masks = []
                
                for result in results:
                    for box_idx, cls in enumerate(result.boxes.cls):
                        if int(cls) in selected_class_ids:
                            try:
                                mask = result.masks.data[box_idx].numpy()
                                mask = cv2.resize(mask, 
                                               (st.session_state['frame_width'],
                                                st.session_state['frame_height']))
                                detected_masks.append(np.where(mask >= 0.3, 1.0, 0))
                            except (IndexError, AttributeError):
                                continue
                
                preview = update_preview(st.session_state['first_frame'], text_sets, detected_masks)
                if preview is not None:
                    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), 
                            use_column_width=True)
            
            # Basic preview second
            with preview_tabs[1]:
                basic_preview = update_preview(st.session_state['first_frame'], text_sets)
                if basic_preview is not None:
                    st.image(cv2.cvtColor(basic_preview, cv2.COLOR_BGR2RGB), 
                            use_column_width=True)
            
            # Process section with better styling
            if video_file and text_sets and selected_classes:
                st.markdown("<hr style='margin: 2rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("Process Video", type="primary", use_container_width=True):
                        job_id = str(uuid.uuid4())
                        with st.spinner("Processing your video..."):
                            success = process_video(
                                video_file,
                                text_sets,
                                selected_classes,
                                model,
                                st.session_state.job_store,
                                job_id,
                                confidence_threshold,
                                iou_threshold
                            )
                            
                            if success:
                                st.success("Processing completed successfully!")
                                status = st.session_state.job_store.get_job_status(job_id)
                                if status.get('result_path'):
                                    with open(status['result_path'], "rb") as file:
                                        st.download_button(
                                            "Download Processed Video",
                                            file,
                                            file_name=f"processed_video.mp4",
                                            mime="video/mp4",
                                            use_container_width=True
                                        )
                            else:
                                st.error("Processing failed. Please try again.")

if __name__ == "__main__":
    main()