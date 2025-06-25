
import os
from pathlib import Path


APP_TITLE = "AI-Powered Indic Image Annotator"
APP_ICON = "🖼️"
PAGE_LAYOUT = "wide"


BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
EXPORTS_DIR = BASE_DIR / "exports"
MODELS_DIR = BASE_DIR / "models"

TEMP_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


SUPPORTED_IMAGE_FORMATS = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp']


BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
YOLO_MODEL_NAME = "yolov5s"


INDIC_LANGUAGES = {
    'Hindi': {
        'code': 'hi',
        'script': 'Devanagari',
        'direction': 'ltr',
        'sample_text': 'यह एक छवि है'
    },
    'Bengali': {
        'code': 'bn',
        'script': 'Bengali',
        'direction': 'ltr',
        'sample_text': 'এটি একটি ছবি'
    },
    'Telugu': {
        'code': 'te',
        'script': 'Telugu',
        'direction': 'ltr',
        'sample_text': 'ఇది ఒక చిత్రం'
    },
    'Marathi': {
        'code': 'mr',
        'script': 'Devanagari',
        'direction': 'ltr',
        'sample_text': 'ही एक प्रतिमा आहे'
    },
    'Tamil': {
        'code': 'ta',
        'script': 'Tamil',
        'direction': 'ltr',
        'sample_text': 'இது ஒரு படம்'
    },
    'Gujarati': {
        'code': 'gu',
        'script': 'Gujarati',
        'direction': 'ltr',
        'sample_text': 'આ એક છબી છે'
    },
    'Kannada': {
        'code': 'kn',
        'script': 'Kannada',
        'direction': 'ltr',
        'sample_text': 'ಇದು ಒಂದು ಚಿತ್ರ'
    },
    'Malayalam': {
        'code': 'ml',
        'script': 'Malayalam',
        'direction': 'ltr',
        'sample_text': 'ഇത് ഒരു ചിത്രമാണ്'
    },
    'Punjabi': {
        'code': 'pa',
        'script': 'Gurmukhi',
        'direction': 'ltr',
        'sample_text': 'ਇਹ ਇੱਕ ਤਸਵੀਰ ਹੈ'
    },
    'Odia': {
        'code': 'or',
        'script': 'Odia',
        'direction': 'ltr',
        'sample_text': 'ଏହା ଏକ ଚିତ୍ର'
    },
    'Urdu': {
        'code': 'ur',
        'script': 'Arabic',
        'direction': 'rtl',
        'sample_text': 'یہ ایک تصویر ہے'
    },
    'English': {
        'code': 'en',
        'script': 'Latin',
        'direction': 'ltr',
        'sample_text': 'This is an image'
    }
}


UI_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#2e8b57',
    'accent': '#ff6b6b',
    'background': '#ffffff',
    'text': '#333333'
}

DEFAULT_SETTINGS = {
    'confidence_threshold': 0.7,
    'max_objects': 10,
    'image_max_width': 800,
    'image_max_height': 600,
    'font_size': 14,
    'annotation_box_color': '#ff6b6b',
    'annotation_text_color': '#333333'
}

EXPORT_FORMATS = {
    'JSON': 'json',
    'CSV': 'csv',
    'COCO': 'coco',
    'YOLO': 'yolo',
    'XML': 'xml'
}


HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', '')
GOOGLE_TRANSLATE_API_KEY = os.getenv('GOOGLE_TRANSLATE_API_KEY', '')


CACHE_TTL = 3600  # 1 hour in seconds
MAX_CACHE_SIZE = 100  # Maximum number of cached items