# GeoTL-IGUIDE

## 🚀 Transfer Learning for Geospatial Analysis on the I-GUIDE Platform

**GeoTL-IGUIDE** is a Jupyter Notebook designed to **run directly on the [I-GUIDE Platform](https://platform.i-guide.io/)**, demonstrating how to perform **Transfer Learning** from **ImageNet pretrained models** to geospatial classification tasks. This notebook bridges the gap between **ImageNet innovations** and **Geospatial Analysis**, providing hands-on guidance for researchers and practitioners.

---

## 🌍 Project Overview

Geospatial data, such as satellite and aerial imagery, presents unique challenges due to limited labeled datasets and domain-specific features. **Transfer Learning** effectively addresses these challenges by:
- Leveraging **ImageNet pretrained models** for geospatial scene classification.
- Reducing training time and computational cost.
- Enhancing model performance through **Feature Extraction** and **Fine-Tuning**.

This notebook demonstrates the application of **Transfer Learning** to geospatial analysis using the **UC Merced Land Use Dataset**, classifying satellite scenes into 21 land use classes, including:
- **Urban**: Medium Residential, Dense Residential, Commercial
- **Agricultural**: Agricultural, Golf Course
- **Natural Scenes**: Forest, Beach, River

---

## 🔑 Key Features

- **Ready-to-Run on I-GUIDE Platform**:
  - Designed to run seamlessly on the **I-GUIDE Platform** with preconfigured dependencies.
- **Three Transfer Learning Approaches**:
  1. **Training from Scratch**: Building a Convolutional Neural Network (CNN) from the ground up.
  2. **Feature Extraction**: Using pretrained models as fixed feature extractors.
  3. **Fine-Tuning**: Adapting pretrained models by unfreezing specific layers.
- **Comprehensive Visualizations**:
  - Visualize accuracy and loss curves to compare model performance.
  - Display predictions alongside true labels for qualitative analysis.

---

## 📚 Dataset Information

This notebook uses the **UC Merced Land Use Dataset**, a benchmark dataset for land use classification, containing:
- **21 classes** including agricultural, urban, forest, and more.
- **100 images per class** with each image of size **256x256 pixels** in the **RGB color space**.

**Dataset Source**: [UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

---

## 🔄 Transfer Learning Approaches

1. **Training from Scratch**:
   - Builds a CNN from the ground up without using any pretrained layers.
   - Suitable when the dataset is large and domain-specific features are needed.

2. **Feature Extraction**:
   - Utilizes pretrained models (e.g., ResNet50) as fixed feature extractors.
   - Only the classifier head is trained, resulting in faster training time.
   - Best for small to medium-sized datasets with similarities to the pretrained dataset.

3. **Fine-Tuning**:
   - Unfreezes the last few layers of the pretrained model to adapt to geospatial imagery.
   - Allows the model to learn more specific features relevant to satellite images.
   - Suitable when the target dataset is larger or significantly different from the pretrained dataset.

---

## ⚙️ Installation and Setup

This notebook is **preconfigured to run directly on the I-GUIDE Platform**. 

All required libraries are automatically installed using the setup_environment_and_files() function:

```python
from util import setup_environment_and_files
setup_environment_and_files()
```



