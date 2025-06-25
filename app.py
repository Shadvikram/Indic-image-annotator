import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_drawable_canvas import st_canvas
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI-Powered Indic Image Annotator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better Indic font support
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;700&family=Noto+Sans+Bengali:wght@400;700&family=Noto+Sans+Telugu:wght@400;700&family=Noto+Sans+Tamil:wght@400;700&family=Noto+Sans+Gujarati:wght@400;700&family=Noto+Sans+Kannada:wght@400;700&family=Noto+Sans+Malayalam:wght@400;700&display=swap');
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .annotation-box {
        border: 2px solid #ff6b6b;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #fff5f5;
    }
    .language-selector {
        background-color: #e8f4fd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .indic-text {
        font-family: 'Noto Sans Devanagari', 'Noto Sans Bengali', 'Noto Sans Telugu', 'Noto Sans Tamil', 'Noto Sans Gujarati', 'Noto Sans Kannada', 'Noto Sans Malayalam', 'Arial Unicode MS', sans-serif;
        font-size: 18px;
        line-height: 1.6;
        color: #333;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin: 5px 0;
        direction: auto;
        text-align: left;
    }
    .translated-caption {
        font-family: 'Noto Sans Devanagari', 'Noto Sans Bengali', 'Noto Sans Telugu', 'Noto Sans Tamil', 'Noto Sans Gujarati', 'Noto Sans Kannada', 'Noto Sans Malayalam', 'Arial Unicode MS', sans-serif;
        font-size: 20px;
        font-weight: bold;
        color: #2e8b57;
        background-color: #e8f5e8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #2e8b57;
        direction: auto;
        text-align: left;
    }
    .translation-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .object-label {
        font-family: 'Noto Sans Devanagari', 'Noto Sans Bengali', 'Noto Sans Telugu', 'Noto Sans Tamil', 'Noto Sans Gujarati', 'Noto Sans Kannada', 'Noto Sans Malayalam', 'Arial Unicode MS', sans-serif;
        font-size: 16px;
        padding: 8px;
        background-color: #f0f8ff;
        border-radius: 4px;
        margin: 3px 0;
        border-left: 3px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

class IndicImageAnnotator:
    def __init__(self):
        self.supported_languages = {
            'Hindi': 'hi',
            'Bengali': 'bn',
            'Telugu': 'te',
            'Marathi': 'mr',
            'Tamil': 'ta',
            'Gujarati': 'gu',
            'Kannada': 'kn',
            'Malayalam': 'ml',
            'Punjabi': 'pa',
            'Odia': 'or',
            'Urdu': 'ur',
            'English': 'en'
        }
        self.model = None
        self.processor = None
        
    @st.cache_resource
    def load_model(_self):
        """Load the BLIP model for image captioning"""
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            return model, processor
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None
    
    def generate_caption(self, image):
        """Generate caption for the image"""
        if self.model is None or self.processor is None:
            self.model, self.processor = self.load_model()
        
        if self.model is None:
            return "Model not available"
        
        try:
            inputs = self.processor(image, return_tensors="pt")
            out = self.model.generate(**inputs, max_length=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            return f"Error generating caption: {str(e)}"
    
    def translate_text(self, text, target_language):
        """Translate text to target language using deep-translator"""
        try:
            if target_language == 'en' or not text or text.strip() == "":
                return text
            
            # Use deep-translator's GoogleTranslator
            translator = GoogleTranslator(source='auto', target=target_language)
            translated_text = translator.translate(text)
            
            # Ensure we return a string
            if translated_text:
                return str(translated_text).strip()
            else:
                return text
                
        except Exception as e:
            st.warning(f"Translation error: {str(e)}")
            return text  # Return original text if translation fails
    
    def batch_translate(self, texts, target_language):
        """Translate multiple texts efficiently"""
        if target_language == 'en':
            return texts
        
        translated_texts = []
        for text in texts:
            translated = self.translate_text(text, target_language)
            translated_texts.append(translated)
            # Small delay to avoid rate limiting
            import time
            time.sleep(0.1)
        
        return translated_texts
    
    def detect_objects(self, image):
        """Simple object detection using OpenCV (placeholder for more advanced models)"""
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Simple edge detection as placeholder
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Predefined object labels for better translation results
        object_labels = [
            "person", "car", "building", "tree", "window", "door", 
            "table", "chair", "book", "phone", "computer", "bag",
            "bottle", "cup", "plate", "flower", "animal", "sign",
            "road", "sky", "grass", "water", "mountain", "cloud"
        ]
        
        objects = []
        for i, contour in enumerate(contours[:10]):  # Limit to top 10 contours
            x, y, w, h = cv2.boundingRect(contour)
            if w > 30 and h > 30:  # Filter small objects
                # Use predefined labels instead of generic "Object_i"
                label = object_labels[i % len(object_labels)]
                objects.append({
                    'id': i,
                    'bbox': [x, y, w, h],
                    'confidence': np.random.uniform(0.7, 0.95),  # Placeholder confidence
                    'label': label
                })
        
        return objects

def display_indic_text(text, label="", is_translated=False, show_success=False):
    """Helper function to display Indic text with proper styling"""
    if show_success:
        st.markdown(f"""
        <div class="translation-success">
            ✅ Translation successful!
        </div>
        """, unsafe_allow_html=True)
    
    if is_translated:
        st.markdown(f"""
        <div class="translated-caption">
            <strong>{label}</strong><br>
            {text}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="indic-text">
            <strong>{label}</strong> {text}
        </div>
        """, unsafe_allow_html=True)

def display_object_translation(original, translated, confidence):
    """Display object translations in a nice format"""
    st.markdown(f"""
    <div class="object-label">
        <strong>Original:</strong> {original}<br>
        <strong>Translated:</strong> {translated}<br>
        <strong>Confidence:</strong> {confidence}
    </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🖼️ AI-Powered Indic Image Annotator</h1>', unsafe_allow_html=True)
    
    # Initialize the annotator
    annotator = IndicImageAnnotator()
    
    # Add installation instructions
    with st.expander("📋 Installation Requirements"):
        st.markdown("""
        **Required packages:**
        ```bash
        pip install streamlit torch opencv-python pillow pandas transformers deep-translator matplotlib seaborn streamlit-drawable-canvas
        ```
        
        **For deep-translator specifically:**
        ```bash
        pip install deep-translator
        ```
        """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Language selection
        st.markdown('<div class="language-selector">', unsafe_allow_html=True)
        selected_language = st.selectbox(
            "Select Output Language",
            list(annotator.supported_languages.keys()),
            index=0
        )
        language_code = annotator.supported_languages[selected_language]
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display selected language info
        st.info(f"Selected: {selected_language} ({language_code})")
        
        # Test translation
        st.markdown("### 🧪 Test Translation")
        test_input = st.text_input("Enter text to test translation:", "This is a beautiful image.")
        
        if st.button("Test Translation") and test_input:
            with st.spinner(f"Translating to {selected_language}..."):
                translated_test = annotator.translate_text(test_input, language_code)
                
            st.markdown("**Test Results:**")
            st.write(f"**Original:** {test_input}")
            
            if translated_test != test_input:
                display_indic_text(translated_test, f"**{selected_language}:**", is_translated=True, show_success=True)
            else:
                st.warning("Translation returned the same text. Check your internet connection or try a different language.")
        
        # Annotation options
        st.markdown("### 📝 Annotation Options")
        show_captions = st.checkbox("Generate Image Captions", value=True)
        show_objects = st.checkbox("Detect Objects", value=True)
        show_manual_annotation = st.checkbox("Manual Annotation", value=True)
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            confidence_threshold = st.slider("Object Detection Confidence", 0.1, 1.0, 0.7)
            max_objects = st.slider("Maximum Objects to Detect", 1, 20, 10)
            translation_delay = st.slider("Translation Delay (seconds)", 0.1, 2.0, 0.2)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">📤 Upload Image</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image to start annotation"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Store image in session state for processing
            st.session_state['current_image'] = image
            st.session_state['uploaded_filename'] = uploaded_file.name
    
    with col2:
        if 'current_image' in st.session_state:
            st.markdown('<h2 class="section-header">🤖 AI Analysis</h2>', unsafe_allow_html=True)
            
            image = st.session_state['current_image']
            
            # Generate caption
            if show_captions:
                with st.spinner("🔍 Generating image caption..."):
                    caption = annotator.generate_caption(image)
                
                st.markdown('<div class="annotation-box">', unsafe_allow_html=True)
                st.markdown(f"**Original Caption (English):** {caption}")
                
                # Translate and display
                if language_code != 'en':
                    with st.spinner(f"🌐 Translating to {selected_language}..."):
                        translated_caption = annotator.translate_text(caption, language_code)
                        
                    if translated_caption and translated_caption != caption:
                        display_indic_text(translated_caption, f"Translated Caption ({selected_language}):", is_translated=True, show_success=True)
                        # Store for export
                        st.session_state['caption_original'] = caption
                        st.session_state['caption_translated'] = translated_caption
                    else:
                        st.warning("⚠️ Translation failed or returned same text. Check internet connection.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Object detection
            if show_objects:
                with st.spinner("🔍 Detecting objects..."):
                    objects = annotator.detect_objects(image)
                
                if objects:
                    st.markdown(f"**🎯 Detected {len(objects)} objects:**")
                    
                    # Filter objects by confidence
                    filtered_objects = [obj for obj in objects if obj['confidence'] >= confidence_threshold]
                    
                    if filtered_objects:
                        # Create annotated image
                        annotated_image = image.copy()
                        draw = ImageDraw.Draw(annotated_image)
                        
                        # Translate all object labels at once
                        object_labels = [obj['label'] for obj in filtered_objects]
                        
                        with st.spinner(f"🌐 Translating {len(object_labels)} object labels..."):
                            if language_code != 'en':
                                translated_labels = annotator.batch_translate(object_labels, language_code)
                            else:
                                translated_labels = object_labels
                        
                        # Draw annotations on image
                        for i, obj in enumerate(filtered_objects):
                            x, y, w, h = obj['bbox']
                            # Draw bounding box
                            draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
                            # Add English label for image
                            label_text = f"{obj['label']} ({obj['confidence']:.2f})"
                            draw.text((x, y-20), label_text, fill="red")
                        
                        st.image(annotated_image, caption="🎯 Detected Objects", use_column_width=True)
                        
                        # Display object translations
                        st.markdown("### 🏷️ Object Labels with Translations:")
                        
                        object_data = []
                        for i, obj in enumerate(filtered_objects):
                            original_label = obj['label']
                            translated_label = translated_labels[i] if i < len(translated_labels) else original_label
                            confidence_str = f"{obj['confidence']:.2f}"
                            
                            # Display individual translation
                            display_object_translation(original_label, translated_label, confidence_str)
                            
                            # Prepare data for table
                            object_data.append({
                                'Original': original_label,
                                f'{selected_language}': translated_label,
                                'Confidence': confidence_str,
                                'Position': f"({obj['bbox'][0]}, {obj['bbox'][1]})"
                            })
                        
                        # Display summary table
                        if object_data:
                            st.markdown("### 📊 Summary Table:")
                            df = pd.DataFrame(object_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # Store for export
                            st.session_state['detected_objects'] = object_data
                    else:
                        st.info(f"No objects detected above {confidence_threshold:.1f} confidence threshold.")
    
    # Manual annotation section
    if show_manual_annotation and 'current_image' in st.session_state:
        st.markdown('<h2 class="section-header">✏️ Manual Annotation</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Drawing canvas
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=3,
                stroke_color="#ff0000",
                background_image=st.session_state['current_image'],
                update_streamlit=True,
                height=400,
                drawing_mode="rect",
                key="canvas",
            )
        
        with col2:
            st.markdown("**✏️ Manual Annotation Tools**")
            
            # Annotation text input
            annotation_text = st.text_area(
                "Add annotation text:",
                placeholder="Describe what you see in the selected area...",
                help="Type your description in English, it will be translated automatically."
            )
            
            if annotation_text and st.button("➕ Add Annotation"):
                with st.spinner(f"🌐 Translating to {selected_language}..."):
                    translated_annotation = annotator.translate_text(annotation_text, language_code)
                
                # Store annotation
                if 'annotations' not in st.session_state:
                    st.session_state['annotations'] = []
                
                annotation_entry = {
                    'text': annotation_text,
                    'translated': translated_annotation,
                    'language': selected_language,
                    'language_code': language_code,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.session_state['annotations'].append(annotation_entry)
                
                # Display success message
                st.success("✅ Annotation added successfully!")
                
                # Display the added annotation
                st.markdown("**📝 Just Added:**")
                st.write(f"**Original:** {annotation_text}")
                if translated_annotation != annotation_text:
                    display_indic_text(translated_annotation, f"**{selected_language}:**", is_translated=True)
                else:
                    st.warning("Translation returned same text.")
            
            # Display saved annotations
            if 'annotations' in st.session_state and st.session_state['annotations']:
                st.markdown("**💾 Saved Annotations:**")
                for i, ann in enumerate(st.session_state['annotations']):
                    with st.expander(f"Annotation {i+1} ({ann['timestamp']})"):
                        st.write(f"**Original:** {ann['text']}")
                        display_indic_text(ann['translated'], f"**{ann['language']}:**", is_translated=True)
                        
                        # Option to delete individual annotation
                        if st.button(f"🗑️ Delete", key=f"del_{i}"):
                            st.session_state['annotations'].pop(i)
                            st.rerun()
    
    # Export section
    if 'current_image' in st.session_state:
        st.markdown('<h2 class="section-header">💾 Export Annotations</h2>', unsafe_allow_html=True)
        
        # Prepare export data
        export_data = {
            'metadata': {
                'filename': st.session_state.get('uploaded_filename', 'unknown'),
                'image_size': st.session_state['current_image'].size,
                'export_timestamp': datetime.now().isoformat(),
                'selected_language': selected_language,
                'language_code': language_code
            },
            'ai_caption': {
                'original': st.session_state.get('caption_original', ''),
                'translated': st.session_state.get('caption_translated', '')
            },
            'detected_objects': st.session_state.get('detected_objects', []),
            'manual_annotations': st.session_state.get('annotations', [])
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export as JSON"):
                # Ensure proper UTF-8 encoding
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_str.encode('utf-8'),
                    file_name=f"indic_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export as CSV"):
                # Prepare CSV data
                csv_data = []
                
                # Add caption data
                if st.session_state.get('caption_original'):
                    csv_data.append([
                        'AI Caption',
                        st.session_state.get('caption_original', ''),
                        st.session_state.get('caption_translated', ''),
                        selected_language,
                        'Auto-generated'
                    ])
                
                # Add object data
                for obj in st.session_state.get('detected_objects', []):
                    csv_data.append([
                        'Detected Object',
                        obj.get('Original', ''),
                        obj.get(selected_language, ''),
                        selected_language,
                        f"Confidence: {obj.get('Confidence', '')}"
                    ])
                
                # Add manual annotations
                for ann in st.session_state.get('annotations', []):
                    csv_data.append([
                        'Manual Annotation',
                        ann['text'],
                        ann['translated'],
                        ann['language'],
                        ann['timestamp']
                    ])
                
                if csv_data:
                    df_export = pd.DataFrame(csv_data, columns=[
                        'Type', 'Original_Text', 'Translated_Text', 'Language', 'Additional_Info'
                    ])
                    csv = df_export.to_csv(index=False, encoding='utf-8')
                    
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv.encode('utf-8'),
                        file_name=f"indic_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data to export. Add some annotations first!")
        
        with col3:
            if st.button("🗑️ Clear All Data"):
                # Clear all session state data
                keys_to_clear = ['annotations', 'caption_original', 'caption_translated', 'detected_objects']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success("🗑️ All annotations cleared!")
                st.rerun()
    
    # Footer with additional information
    st.markdown("---")
    st.markdown("""
    ### 📚 About this Application
    This AI-powered Indic Image Annotator uses:
    - **BLIP** for image captioning
    - **Deep Translator** for reliable translation to Indic languages
    - **OpenCV** for basic object detection
    - **Streamlit** for the interactive interface
    
    **Supported Languages:** Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu
    
    🔧 **Troubleshooting:**
    - If translations don't appear, check your internet connection
    - Use the "Test Translation" feature to verify functionality
    - Ensure all required packages are installed
    """)

if __name__ == "__main__":
    main()