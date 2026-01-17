# 🚗 AI Car Damage Assessment Tool

**An intelligent end-to-end computer vision system for automated vehicle damage inspection.**

## 📸 Model Capabilities

The system automatically triages vehicle images into three categories of analysis.

| **Frontal Damage** | **Rear Damage** | **No Damage** |
|:---:|:---:|:---:|
| ![Front Crush](predicted%20images/front_crush.png) | ![Rear Crush](predicted%20images/rear_crush.png) | ![Normal Car](predicted%20images/normal.png) |
| *Detected: Front | Severe* | *Detected: Rear | Moderate* | *Detected: Whole (Undamaged)* |

## 📌 Project Overview

Manual car inspection for insurance claims is time-consuming and subjective. This project provides an automated AI solution that assesses vehicle condition from a single image.

The application uses a **DenseNet121** based Convolutional Neural Network (CNN) to perform a three-stage classification pipeline:
1.  **Gate Classification:** Determines if the car is "Damaged" or "Not Damaged" (Whole).
2.  **Location Detection:** If damaged, identifies the area: `Front`, `Rear`, or `Side`.
3.  **Severity Estimation:** Assesses the extent of damage: `Minor`, `Moderate`, or `Severe`.

## 🛠️ Architecture & Logic

The system is modularized into a web interface and an inference engine:

* **`app.py`:** The Streamlit frontend that handles user image uploads and displays results.
* **`model_helper.py`:** Contains the `CarDamagePredictor` class. It handles:
    * **Image Preprocessing:** Resizing (224x224), Center Cropping, and Normalization.
    * **Model Loading:** Loads the consolidated weights (`gate`, `location`, `severity`) from `car_damage_pred_model.pth`.
    * **Inference:** Runs the image through the respective model heads based on the pipeline logic.

### Image Processing Pipeline
To ensure prediction consistency, images undergo the following transformations before inference:
1.  **Resize**: Images are scaled to $224 \times 224$ pixels.
2.  **Normalize**: Standard ImageNet normalization is applied with $\mu=[0.485, 0.456, 0.406]$ and $\sigma=[0.229, 0.224, 0.225]$.
3.  **Tensor Conversion**: Data is converted into PyTorch tensors and batched for the model.

## 📊 Supported Damage Classes
The model is trained to recognize and categorize vehicle images into one of the following **6 classes**:

* **Front Breakage**: Identifies structural snaps or cracks in the front bumper, grille, or headlights.
* **Front Crushed**: Detects severe impact deformation or "crunched" areas at the front of the vehicle.
* **Front Normal**: Confirms the front section of the vehicle is intact and free of visible damage.
* **Rear Breakage**: Identifies cracks, snaps, or fragments missing from the rear bumper or taillights.
* **Rear Crushed**: Detects significant collision impact or caved-in sections at the rear.
* **Rear Normal**: Confirms the rear section of the vehicle is in its original, undamaged state.

## 📂 File Structure

* **`app.py`**: The main Streamlit frontend for file uploads and displaying classification results.
* **`model_helper.py`**: Contains the `CarClassifierRestNet` class definition and the logic for pre-processing and prediction.
* **`car_damage_pred_model.pth`**: The saved state dictionary of the trained ResNet-50 model.

## 🚀 How to Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/AI-Car-Damage-Assessment.git](https://github.com/yourusername/AI-Car-Damage-Assessment.git)
    cd AI-Car-Damage-Assessment
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Requires `torch`, `torchvision`, `streamlit`, `pillow`)*

3.  **Launch the App**
    ```bash
    streamlit run app.py
    ```

4.  **Use the Tool**
    * Open the local URL provided (usually `http://localhost:8501`).
    * Upload an image of a car (.jpg, .jpeg, .png).
    * View the automated assessment report.

## 📂 File Structure

* `app.py`: Main application script.
* `model_helper.py`: Model architecture and inference logic.
* `car_damage_pred_model.pth`: Pre-trained PyTorch model weights.
* `requirements.txt`: Python dependencies.
* `demo_*.png`: Demo
