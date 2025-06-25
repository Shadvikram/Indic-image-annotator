# 🖼️ AI-Powered Indic Image Annotator

A comprehensive Streamlit application for annotating images with AI assistance, supporting multiple Indic languages.

## ✨ Features

- **Multi-language Support**: 12 Indic languages + English
- **AI-Powered Captions**: Automatic image captioning using BLIP model
- **Object Detection**: Automatic object detection and labeling
- **Manual Annotation**: Interactive annotation tools
- **Translation**: Real-time translation to Indic languages
- **Export Options**: JSON, CSV, COCO, YOLO formats
- **Visual Analytics**: Annotation statistics and insights

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Internet connection (for model downloads)

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd indic-image-annotator
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```

3. **Start the application**
   ```bash
   # Option 1: Use run script
   ./run.sh          # Linux/Mac
   run.bat           # Windows
   
   # Option 2: Direct command
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
indic-image-annotator/
├── app.py                 # Main Streamlit application
├── utils.py              # Utility functions
├── config.py             # Configuration settings
├── setup.py              # Setup script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── run.sh               # Linux/Mac run script
├── run.bat              # Windows run script
├── .gitignore           # Git ignore file
├── temp/                # Temporary files
├── exports/             # Exported annotations
├── models/              # Downloaded AI models
└── sample_images/       # Sample images for testing
```

## 🎯 How to Use

### 1. Upload Image
- Click "Choose an image file" in the sidebar
- Supported formats: PNG, JPG, JPEG, BMP, TIFF, WEBP

### 2. Select Language
- Choose your preferred output language from the dropdown
- Supports 12 Indic languages + English

### 3. AI Analysis
- **Auto Captions**: Get AI-generated descriptions
- **Object Detection**: Identify objects in the image
- **Translation**: View results in your selected language

### 4. Manual Annotation
- Use drawing tools to mark regions
- Add text annotations
- Save annotations with timestamps

### 5. Export Results
- Download annotations in multiple formats
- JSON, CSV, COCO, YOLO supported
- Include translation data

## 🔧 Configuration

### Language Settings
Edit `config.py` to modify supported languages:
```python
INDIC_LANGUAGES = {
    'Hindi': {'code': 'hi', 'script': 'Devanagari'},
    # Add more languages...
}
```

### Model Settings
Configure AI models in `config.py`:
```python
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
YOLO_MODEL_NAME = "yolov5s"
```

### UI Customization
Modify colors and layout:
```python
UI_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#2e8b57',
    # Customize colors...
}
```

## 🤖 Supported AI Models

- **BLIP**: Image captioning and visual question answering
- **YOLO**: Object detection and localization
- **Google Translate**: Multi-language translation
- **CLIP**: Image-text understanding (optional)

## 🌐 Supported Languages

| Language  | Script      | Code |
|-----------|-------------|------|
| Hindi     | Devanagari  | hi   |
| Bengali   | Bengali     | bn   |
| Telugu    | Telugu      | te   |
| Marathi   | Devanagari  | mr   |
| Tamil     | Tamil       | ta   |
| Gujarati  | Gujarati    | gu   |
| Kannada   | Kannada     | kn   |
| Malayalam | Malayalam   | ml   |
| Punjabi   | Gurmukhi    | pa   |
| Odia      | Odia        | or   |
| Urdu      | Arabic      | ur   |
| English   | Latin       | en   |

## 📊 Export Formats

### JSON Format
```json
{
  "image_info": {...},
  "language": "Hindi",
  "ai_caption": "...",
  "detected_objects": [...],
  "manual_annotations": [...]
}
```

### COCO Format
Standard COCO dataset format with additional language fields.

### YOLO Format
Bounding box coordinates in YOLO format.

### CSV Format
Tabular data with annotations and translations.

## 🔍 Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check internet connection
   - Models will download on first use
   - Manually download: `python -c "from transformers import BlipProcessor; BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')"`

2. **Translation Errors**
   - Google Translate API limits may apply
   - Check network connectivity
   - Some languages may have limited support

3. **Memory Issues**
   - Reduce image size for large files
   - Close other applications if running out of RAM
   - Consider using CPU instead of GPU for inference

4. **Port Already in Use**
   - Change port: `streamlit run app.py --server.port 8502`
   - Kill existing processes: `pkill -f streamlit`

### Performance Tips

- **GPU Acceleration**: Install CUDA for faster inference
- **Image Optimization**: Resize large images before processing
- **Batch Processing**: Process multiple images efficiently
- **Caching**: Models are cached for faster subsequent runs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black app.py utils.py config.py

# Type checking
mypy app.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For providing pre-trained models
- **Streamlit**: For the amazing web framework
- **Google Translate**: For translation services
- **OpenCV**: For image processing capabilities
- **Community**: For feedback and contributions

## 📞 Support

- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join community discussions
- **Documentation**: Check the wiki for detailed guides
- **Email**: Contact the maintainers

## 🗺️ Roadmap

- [ ] Video annotation support
- [ ] Advanced object tracking
- [ ] Custom model training
- [ ] API endpoints
- [ ] Mobile app integration
- [ ] Collaborative annotation
- [ ] Advanced analytics dashboard

---

Made with ❤️ for the Indic language community