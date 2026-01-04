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

### preprocessing.ipynb
- Loads raw tabular data
- Handles missing values and invalid entries
- Performs feature selection and basic feature engineering
- Scales numerical features for model compatibility
- Prepares cleaned tabular datasets for modeling
- downloads the images by calling API

### feature_extraction_image.ipynb
- Loads downloaded satellite images
- Applies image preprocessing (resizing, normalization)
- Extracts numerical feature representations from images(RESNET50)
- Converts image data into machine-learning-ready feature vectors
- Saves extracted image features for downstream modeling

### tabular_model_training.ipynb
- Trains regression models on Tabular features only
- Evaluates model performance using standard regression metrics

### multimodel_training.ipyng
- Merges both image and tabular data
- Trains the model on merged data
- Do the final prediction and saves the .csv file

---

## Dataset Description

### Tabular Features
- Bedrooms, bathrooms, living area (`sqft_living`), and lot size (`sqft_lot`)
- Number of floors, property condition, and grade
- Waterfront and view indicators
- Nearby property statistics (`sqft_living15`, `sqft_lot15`)
- **Target variable:** `price`, transformed using `log(price)` for numerical stability

### Image Data
- Satellite images corresponding to each property location
- Capture environmental and spatial context such as greenery, water bodies, and urban or road density

---

## Modeling Approach

### 1. Tabular Data Model
- Missing value imputation, feature scaling
- Trained multiple Regressor models using only structured data
  
### 2. Image Feature Extraction
- CNN Backbone: ResNet-50 (pretrained on ImageNet)
- Extracted high dimensional image features
  
### 3. Multimodal Fusion
- Concatenated tabular features with image embeddings
- Train multiple models on merged data
- Taken the best model out of those models to make the predictions

---

## How to Run

### Clone the Repository
```bash
git clone https://https://github.com/ankitdhanghar/Satellite-Imagery-Based-Property-Valuation.git
cd Satellite-Imagery-Based-Property-Valuation
```

### Create virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirement.txt
```

### download satellite images
 ```bash
python data_fetcher.py
```

### Run Jupyter Notebooks in order
```bash
preprocessing.ipynb > feature_extraction_image.ipynb > multimodal_training.ipynb
```

---

## Property Price Prediction Models

This project compares the performance of different models for predicting property prices.  

| Model                        | R² Score |
|-------------------------------|----------|
| Tabular Only                  | Moderate |
| Tabular + Satellite Imagery   | Higher   |

**Summary:**  
The multimodal model (Tabular + Satellite Imagery) outperformed the baseline tabular-only model, demonstrating that visual contextual cues significantly improve property price predictions.

---
## Tech Stack

- **Data Handling:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Deep Learning:** PyTorch, torchvision  
- **Image Processing:** PIL, OpenCV  
- **Visualization:** Matplotlib, Seaborn

---

## Author

**Ankit Dhanghar**














