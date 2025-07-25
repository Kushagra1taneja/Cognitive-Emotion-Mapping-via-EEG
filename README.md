# EEG Emotion Analysis Project

This project is a Streamlit web application for emotion classification based on EEG (Electroencephalogram) data. It utilizes machine learning and deep learning models to predict emotional states (Positive, Negative, Neutral) from preprocessed EEG signals.

![App Screenshot](images/screenshot.png)

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Deployment](#deployment)

## Features

-   **Interactive Web Interface**: Built with Streamlit for easy interaction.
-   **Multiple Model Predictions**: Uses both a Neural Network and a Random Forest model for robust predictions.
-   **File Upload**: Users can upload their own EEG data in CSV format.
-   **Sample Data**: Includes sample data for a quick demonstration.
-   **Data Visualization**: Provides visualizations of prediction distributions and model performance metrics.
-   **Performance Metrics**: Displays accuracy, confusion matrices, and classification reports if the uploaded data has labels.

## Project Structure

The project is organized as follows:

```
EEG_Emotion_Analysis_Project/
│
├── .streamlit/
│   └── config.toml         # Streamlit configuration (optional)
├── models/
│   ├── encoder_model.h5
│   ├── final_selected_features.joblib
│   ├── nn_model.h5
│   ├── rf_model.joblib
│   └── scaler.joblib
├── images/
│   └── screenshot.png      # Add app screenshots here
│   └── graph1.png          # Add graph images here
│   └── graph2.png
├── app.py                  # Main Streamlit application file
├── requirements.txt        # Python dependencies
├── emotions.csv            # Dataset used for training
├── sample_input_test.csv   # Sample data for testing the app
└── README.md               # This file
```

## Dataset

The dataset (`emotions.csv`) contains preprocessed EEG data. Each row represents a sample with various features extracted from EEG signals, and a label corresponding to an emotional state.

## Models

Two models are used for prediction:

1.  **Neural Network**: A deep learning model with an autoencoder for feature reduction.
2.  **Random Forest**: An ensemble learning method known for its high accuracy and robustness.

The models were trained on the `emotions.csv` dataset. The training process is detailed in the `EEG_Emotion_Analysis.ipynb` notebook (not included in the deployed app for brevity).

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

-   Python 3.8+
-   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/EEG_Emotion_Analysis_Project.git
    cd EEG_Emotion_Analysis_Project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application locally, use the following command:

```bash
streamlit run app.py
```

Your web browser should open with the application running at `http://localhost:8501`.

## Screenshots

Here are some screenshots of the application.

**Main Dashboard:**
![App Screenshot](images/screenshot.png)

**Prediction Distribution:**
![Graph 1](images/graph1.png)

**Model Performance:**
![Graph 2](images/graph2.png)

*(You can replace the placeholder images in the `images/` directory with your own screenshots.)*

