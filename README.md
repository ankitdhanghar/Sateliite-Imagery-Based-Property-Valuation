# Satellite-Imagery-Based-Property-Valuation
## Project Overview

This project develops a **multimodal regression system** that integrates **tabular data** with **satellite imagery** to predict a target variable more accurately than traditional tabular-only approaches.

While structured features capture explicit numerical and categorical information, they often miss **spatial and environmental context** such as surrounding infrastructure, land usage, and neighborhood patterns. Satellite images provide this complementary visual information.

The project implements a complete end-to-end pipeline including:
- Satellite image collection using geospatial coordinates
- Tabular data preprocessing and feature engineering
- Image feature extraction
- Fusion of image and tabular features
- Regression model training and evaluation


---
##  Notebook Description

### data_fetcher.py
- Handles satellite image collection using the Mapbox Static Images API based on latitude and longitude coordinates.  
- Downloads and saves images locally in JPEG format with configurable size and zoom level.  
- Manages API errors and validates successful image retrieval.  
- Generates a CSV index linking record IDs to image file paths and availability status.

---

### preprocessing.ipynb
- Loads raw tabular data
- Handles missing values and invalid entries
- Performs feature selection and basic feature engineering
- Scales numerical features for model compatibility
- Prepares cleaned tabular datasets for modeling
- downloads the images by calling API

---

### feature_extraction_image.ipynb
- Loads downloaded satellite images
- Applies image preprocessing (resizing, normalization)
- Extracts numerical feature representations from images(RESNET50)
- Converts image data into machine-learning-ready feature vectors
- Saves extracted image features for downstream modeling

---

### tabular_model_training.ipynb
- Trains regression models on Tabular features only
- Evaluates model performance using standard regression metrics

### multimodel_training.ipyng
- Merges both image and tabular data
- Trains the model on merged data
- Do the final prediction and saves the .csv file

  ---




