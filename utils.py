import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
from googletrans import Translator
import streamlit as st
import requests
from io import BytesIO
import base64
import pandas as pd
class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    def resize_image(image, max_width=800, max_height=600):
        """Resize image while maintaining aspect ratio"""
        width, height = image.size
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    @staticmethod
    def enhance_image(image, brightness=1.0, contrast=1.0, saturation=1.0):
        """Enhance image with brightness, contrast, and saturation adjustments"""
        from PIL import ImageEnhance
        
        # Brightness
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        
        # Contrast
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        
        # Saturation
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
        
        return image
    
    @staticmethod
    def apply_filters(image, filter_type="none"):
        """Apply various filters to the image"""
        img_array = np.array(image)
        
        if filter_type == "blur":
            img_array = cv2.GaussianBlur(img_array, (15, 15), 0)
        elif filter_type == "sharpen":
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img_array = cv2.filter2D(img_array, -1, kernel)
        elif filter_type == "edge":
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            img_array = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        elif filter_type == "sepia":
            sepia_filter = np.array([[0.393, 0.769, 0.189],
                                   [0.349, 0.686, 0.168],
                                   [0.272, 0.534, 0.131]])
            img_array = cv2.transform(img_array, sepia_filter)
            img_array = np.clip(img_array, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))

class LanguageManager:
    """Manage language operations and translations"""
    
    def __init__(self):
        self.translator = Translator()
        self.indic_languages = {
            'Hindi': {'code': 'hi', 'script': 'Devanagari', 'font': 'NotoSansDevanagari'},
            'Bengali': {'code': 'bn', 'script': 'Bengali', 'font': 'NotoSansBengali'},
            'Telugu': {'code': 'te', 'script': 'Telugu', 'font': 'NotoSansTelugu'},
            'Marathi': {'code': 'mr', 'script': 'Devanagari', 'font': 'NotoSansDevanagari'},
            'Tamil': {'code': 'ta', 'script': 'Tamil', 'font': 'NotoSansTamil'},
            'Gujarati': {'code': 'gu', 'script': 'Gujarati', 'font': 'NotoSansGujarati'},
            'Kannada': {'code': 'kn', 'script': 'Kannada', 'font': 'NotoSansKannada'},
            'Malayalam': {'code': 'ml', 'script': 'Malayalam', 'font': 'NotoSansMalayalam'},
            'Punjabi': {'code': 'pa', 'script': 'Gurmukhi', 'font': 'NotoSansGurmukhi'},
            'Odia': {'code': 'or', 'script': 'Odia', 'font': 'NotoSansOriya'},
            'Urdu': {'code': 'ur', 'script': 'Arabic', 'font': 'NotoSansArabic'},
            'English': {'code': 'en', 'script': 'Latin', 'font': 'Arial'}
        }
    
    def translate_text(self, text, target_language_code, source_language='auto'):
        """Translate text with error handling"""
        try:
            if target_language_code == 'en' and source_language == 'auto':
                return text
            
            result = self.translator.translate(text, src=source_language, dest=target_language_code)
            return result.text
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            return text
    
    def detect_language(self, text):
        """Detect the language of the given text"""
        try:
            result = self.translator.detect(text)
            return result.lang
        except Exception as e:
            st.error(f"Language detection error: {str(e)}")
            return 'en'
    
    def get_language_info(self, language_name):
        """Get language information"""
        return self.indic_languages.get(language_name, self.indic_languages['English'])

class AnnotationManager:
    """Manage annotations and export functionality"""
    
    @staticmethod
    def create_annotation_overlay(image, annotations, font_size=20):
        """Create an overlay with annotations on the image"""
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        
        # Try to load appropriate font
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        for i, annotation in enumerate(annotations):
            # Draw annotation box
            x, y = annotation.get('position', (10, 10 + i * 30))
            text = annotation.get('text', '')
            
            # Draw background rectangle
            bbox = draw.textbbox((x, y), text, font=font)
            draw.rectangle(bbox, fill='white', outline='black', width=2)
            
            # Draw text
            draw.text((x + 5, y + 5), text, fill='black', font=font)
        
        return overlay
    
    @staticmethod
    def export_to_coco_format(annotations, image_info):
        """Export annotations in COCO format"""
        coco_data = {
            "images": [{
                "id": 1,
                "file_name": image_info.get('filename', 'image.jpg'),
                "width": image_info.get('width', 0),
                "height": image_info.get('height', 0)
            }],
            "annotations": [],
            "categories": []
        }
        
        category_id = 1
        categories = {}
        
        for i, annotation in enumerate(annotations):
            category_name = annotation.get('category', 'object')
            
            if category_name not in categories:
                categories[category_name] = category_id
                coco_data['categories'].append({
                    "id": category_id,
                    "name": category_name
                })
                category_id += 1
            
            bbox = annotation.get('bbox', [0, 0, 0, 0])  # [x, y, width, height]
            
            coco_data['annotations'].append({
                "id": i + 1,
                "image_id": 1,
                "category_id": categories[category_name],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "attributes": {
                    "text": annotation.get('text', ''),
                    "translated_text": annotation.get('translated_text', ''),
                    "language": annotation.get('language', 'en')
                }
            })
        
        return coco_data
    
    @staticmethod
    def export_to_yolo_format(annotations, image_info):
        """Export annotations in YOLO format"""
        yolo_data = []
        image_width = image_info.get('width', 1)
        image_height = image_info.get('height', 1)
        
        for annotation in annotations:
            bbox = annotation.get('bbox', [0, 0, 0, 0])  # [x, y, width, height]
            
            # Convert to YOLO format (normalized center coordinates)
            x_center = (bbox[0] + bbox[2] / 2) / image_width
            y_center = (bbox[1] + bbox[3] / 2) / image_height
            width = bbox[2] / image_width
            height = bbox[3] / image_height
            
            class_id = 0  # Default class
            
            yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return '\n'.join(yolo_data)

class ModelManager:
    """Manage different AI models for image annotation"""
    
    @staticmethod
    @st.cache_resource
    def load_clip_model():
        """Load CLIP model for image-text understanding"""
        try:
            import clip
            model, preprocess = clip.load("ViT-B/32")
            return model, preprocess
        except ImportError:
            st.warning("CLIP model not available. Install with: pip install git+https://github.com/openai/CLIP.git")
            return None, None
    
    @staticmethod
    @st.cache_resource
    def load_yolo_model():
        """Load YOLO model for object detection"""
        try:
            import torch
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            return model
        except Exception as e:
            st.warning(f"YOLO model not available: {str(e)}")
            return None
    
    @staticmethod
    def run_yolo_inference(model, image):
        """Run YOLO inference on image"""
        if model is None:
            return []
        
        try:
            results = model(image)
            detections = results.pandas().xyxy[0].to_dict('records')
            
            objects = []
            for detection in detections:
                objects.append({
                    'label': detection['name'],
                    'confidence': detection['confidence'],
                    'bbox': [
                        int(detection['xmin']),
                        int(detection['ymin']),
                        int(detection['xmax'] - detection['xmin']),
                        int(detection['ymax'] - detection['ymin'])
                    ]
                })
            
            return objects
        except Exception as e:
            st.error(f"YOLO inference error: {str(e)}")
            return []

class DataAnalyzer:
    """Analyze annotation data and generate insights"""
    
    @staticmethod
    def analyze_annotations(annotations):
        """Analyze annotation data and return insights"""
        if not annotations:
            return {}
        
        # Language distribution
        languages = [ann.get('language', 'Unknown') for ann in annotations]
        language_counts = pd.Series(languages).value_counts()
        
        # Text length analysis
        text_lengths = [len(ann.get('text', '')) for ann in annotations]
        
        # Timestamp analysis
        timestamps = [ann.get('timestamp', '') for ann in annotations if ann.get('timestamp')]
        
        analysis = {
            'total_annotations': len(annotations),
            'language_distribution': language_counts.to_dict(),
            'avg_text_length': np.mean(text_lengths) if text_lengths else 0,
            'text_length_std': np.std(text_lengths) if text_lengths else 0,
            'unique_languages': len(language_counts),
            'timestamps': timestamps
        }
        
        return analysis
    
    @staticmethod
    def create_visualization(analysis_data):
        """Create visualizations for annotation analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Language distribution pie chart
        if analysis_data.get('language_distribution'):
            languages = list(analysis_data['language_distribution'].keys())
            counts = list(analysis_data['language_distribution'].values())
            
            axes[0, 0].pie(counts, labels=languages, autopct='%1.1f%%')
            axes[0, 0].set_title('Language Distribution')
        
        # Text length histogram
        if 'text_lengths' in analysis_data:
            axes[0, 1].hist(analysis_data['text_lengths'], bins=10, edgecolor='black')
            axes[0, 1].set_title('Text Length Distribution')
            axes[0, 1].set_xlabel('Character Count')
            axes[0, 1].set_ylabel('Frequency')
        
        # Timeline (if timestamps available)
        if analysis_data.get('timestamps'):
            try:
                timestamps = pd.to_datetime(analysis_data['timestamps'])
                axes[1, 0].plot(timestamps, range(len(timestamps)), marker='o')
                axes[1, 0].set_title('Annotation Timeline')
                axes[1, 0].set_xlabel('Time')
                axes[1, 0].set_ylabel('Annotation Count')
            except:
                axes[1, 0].text(0.5, 0.5, 'Timeline not available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Summary statistics
        stats_text = f"""
        Total Annotations: {analysis_data.get('total_annotations', 0)}
        Unique Languages: {analysis_data.get('unique_languages', 0)}
        Avg Text Length: {analysis_data.get('avg_text_length', 0):.1f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()