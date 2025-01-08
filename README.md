# textbehindvideo

# Video Text Overlay

A powerful application for adding intelligent text overlays to videos with object detection, developed by [Shotcut.in](https://shotcut.in).

## 🌟 Overview

Video Text Overlay is a sophisticated web application that combines state-of-the-art object detection with professional text overlay capabilities. Built with Streamlit, it offers an intuitive interface for adding text to videos while respecting the presence of detected objects in the scene.

## ✨ Key Features

- **Intelligent Text Overlays**: Smart text placement that interacts with detected objects
- **Multi-layer Text Support**: Create unlimited text layers with independent styling
- **Advanced Object Detection**: Powered by YOLOv8 segmentation
- **Real-time Preview**: Instant visualization of your text overlays
- **Professional Text Styling**:
  - Complete Merriweather font family support
  - Full control over size, color, opacity
  - Precise positioning and rotation
  - Multiple font weights
- **Flexible Detection Settings**: Adjustable confidence and IOU thresholds

## 🛠️ Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd video-text-overlay
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLOv8 model:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt
```

4. Set up fonts:
   - Download Merriweather font family
   - Place font files in the `fonts/` directory

## 🚀 Usage

1. Launch the application:
```bash
streamlit run app.py
```
or 
```bash
python -m streamlit run app.py
```

2. Follow the intuitive UI:
   - Upload your video
   - Select objects to detect
   - Add and customize text layers
   - Process and download the final video

## 🙏 Acknowledgments

This project stands on the shoulders of giants. We're grateful to:

- **YOLOv8**: For state-of-the-art object detection and segmentation
- **Streamlit**: For the powerful web application framework
- **OpenCV**: For comprehensive video processing capabilities
- **Pillow**: For advanced image processing and text rendering
- **NumPy**: For efficient numerical operations
- **Merriweather**: For the beautiful font family

## 📝 License

MIT License

Copyright (c) 2024 Shotcut.in

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## 🔧 Technical Details

### Dependencies
- Python 3.8+
- Streamlit >= 1.24.0
- YOLOv8 (via Ultralytics)
- OpenCV >= 4.7.0
- Pillow >= 9.5.0
- NumPy >= 1.24.0

### Object Detection
- Uses YOLOv8 segmentation model
- Supports 80 object classes
- Real-time object detection and masking
- Configurable confidence thresholds

### Text Processing
- PIL-based text rendering
- Advanced styling options
- Multiple layer support
- Real-time preview capabilities

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📫 Support

For questions, feature requests, or support:
- Create an issue in the repository
- Contact [support@shotcut.in](mailto:support@shotcut.in)
- Visit [shotcut.in](https://shotcut.in) for more information

## 🛣️ Roadmap

- [ ] Additional font family support
- [ ] Custom object detection model support
- [ ] Advanced animation effects
- [ ] Batch processing capabilities
- [ ] Export in multiple formats

Made with ❤️ by Shotcut.in
