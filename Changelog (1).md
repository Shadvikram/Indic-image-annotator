# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] 

### Added

- **Visual analytics dashboard**: Introduced a new dashboard interface providing insights into image annotations, such as label distribution, annotation frequency, and contributor activity.
- **YOLO export format**: Added the ability to export annotation data in YOLO format for compatibility with training pipelines using the YOLO object detection framework.
- **Sample images**: Included a set of curated sample images located in the `sample_images/` directory for demo and testing purposes.
- **Language support for Urdu**: Expanded language options by adding full support for Urdu (language code: `ur`), including translation and text rendering.

### Changed

- **BLIP model update**: Replaced the previous BLIP image captioning model with `Salesforce/blip-image-captioning-base` to improve caption accuracy and performance.
- **UI improvements**: Enhanced overall responsiveness and accessibility of the Streamlit UI, including better support for screen readers and high-contrast mode.

### Fixed

- **Indic script translation display**: Resolved an issue where translations in certain Indic scripts were not displaying properly due to font and encoding limitations.
- **TIFF image preview**: Fixed a bug that prevented TIFF-format images from being previewed correctly in the Streamlit app.

---

## [1.1.0]

### Added

- **Manual annotation tools**: Enabled users to manually draw bounding boxes and label objects on uploaded images within the UI.
- **Language dropdown**: Introduced a dropdown menu supporting translation and UI display for 12 Indic languages plus English.
- **Additional export formats**: Added support for exporting annotations in JSON, CSV, and COCO formats for broader compatibility with ML workflows.

### Changed

- **UI layout**: Redesigned the layout using collapsible sections to improve organization and streamline the user experience.
- **Browser compatibility**: Enhanced support for older browsers by optimizing frontend assets and fallback behavior.

### Fixed

- **Initial caption display bug**: Resolved an issue where the AI-generated caption was not displayed on image upload.
- **Bounding box format**: Corrected formatting issues in the exported bounding boxes for object detection tasks.

---

## [1.0.0]

### Added

- **Initial release** with core features:
  - **AI-powered image captioning** using the BLIP model for generating descriptive captions of uploaded images.
  - **Object detection** integrated with YOLOv5 for identifying and labeling multiple objects in images.
  - **Real-time translation** of captions and labels using Google Translate API.
  - **Streamlit frontend** allowing users to upload images, view results, and interact with tools in a clean UI.

---


