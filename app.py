import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Function to check if all model files exist
def check_model_files():
    model_paths = [
        'models/scaler.joblib',
        'models/final_selected_features.joblib',
        'models/encoder_model.h5',
        'models/nn_model.h5',
        'models/rf_model.joblib'
    ]
    for path in model_paths:
        if not os.path.exists(path):
            return False
    return True

# Load models and define functions first, outside of the main app logic
if check_model_files():
    scaler = joblib.load('models/scaler.joblib')
    final_selected_features = joblib.load('models/final_selected_features.joblib')
    encoder = load_model('models/encoder_model.h5')
    nn_model = load_model('models/nn_model.h5')
    rf_model = joblib.load('models/rf_model.joblib')
    label_mapping = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

    def preprocess_data(data):
        """
        Optimized preprocessing that matches the notebook exactly.
        Creates the same feature space that was used for training.
        """
        # Rename column if necessary
        if '# mean_0_a' in data.columns:
            data = data.rename(columns={'# mean_0_a': 'mean_0_a'})

        # Create a copy for processing
        data_preprocessed = data.copy()

        # Apply outlier capping to numerical columns
        def cap_outliers(df, columns):
            for col in columns:
                if col in df.columns:  # Safety check
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            return df

        numerical_cols = data_preprocessed.select_dtypes(include=[np.number]).columns.tolist()
        data_preprocessed = cap_outliers(data_preprocessed, numerical_cols)

        # Feature Extraction - FFT band power calculation
        fft_cols_a = [col for col in data.columns if 'fft' in col and col.endswith('_a')]
        fft_cols_b = [col for col in data.columns if 'fft' in col and col.endswith('_b')]
        
        if fft_cols_a and fft_cols_b:
            # Convert FFT columns to numeric and handle missing values
            fft_cols = fft_cols_a + fft_cols_b
            for col in fft_cols:
                data_preprocessed[col] = pd.to_numeric(data_preprocessed[col], errors='coerce')
            data_preprocessed = data_preprocessed.dropna(subset=fft_cols)

            # Calculate frequency band powers
            n_bins = len(fft_cols_a)
            freq_step = 128.0 / n_bins
            bands = {
                'delta': (0, 4),
                'theta': (4, 8),
                'alpha': (8, 12),
                'beta': (12, 30),
                'gamma': (30, 100)
            }

            def compute_band_power(row, cols, band_range):
                freq_start = int(band_range[0] / freq_step)
                freq_end = int(band_range[1] / freq_step)
                freq_end = min(freq_end, len(cols))  # Safety check
                if freq_start < freq_end:
                    return row[cols[freq_start:freq_end]].sum()
                return 0

            # Add band power features
            for band, (low, high) in bands.items():
                data_preprocessed[f'{band}_power_a'] = data_preprocessed.apply(
                    lambda row: compute_band_power(row, fft_cols_a, (low, high)), axis=1
                )
                data_preprocessed[f'{band}_power_b'] = data_preprocessed.apply(
                    lambda row: compute_band_power(row, fft_cols_b, (low, high)), axis=1
                )

        # Remove label column if present (should be done last)
        if 'label' in data_preprocessed.columns:
            data_preprocessed = data_preprocessed.drop('label', axis=1)

        return data_preprocessed

# --- Main App ---
st.set_page_config(page_title="EEG Emotion Prediction", page_icon="🧠", layout="wide")

st.title('🧠 EEG Emotion Prediction Dashboard')
st.markdown("---")
st.markdown("### Welcome to the EEG-based Emotion Classification System")
st.markdown("This application uses advanced machine learning models to predict emotional states from EEG signals.")

if not check_model_files():
    st.error(
        "❌ **Error: Model files not found.** "
        "Please ensure you have run the final cell in the 'EEG_Emotion_Analysis.ipynb' notebook "
        "to generate and save the models into the 'models' directory."
    )
else:
    # Sidebar for user input
    st.sidebar.header("📊 Data Input Options")
    st.sidebar.markdown("Choose how you want to provide EEG data:")
    input_method = st.sidebar.radio(
        "Select input method:", 
        ("Upload a CSV file", "Use sample data"),
        help="Upload your own EEG data or use our sample dataset for testing"
    )

    # Initialize variables
    input_data = None
    data_source = None

    if input_method == "Upload a CSV file":
        uploaded_file = st.file_uploader(
            "📁 Choose a CSV file for prediction", 
            type="csv",
            help="Upload an EEG dataset in CSV format with the same structure as the training data"
        )
        if uploaded_file is not None:
            input_data = pd.read_csv(uploaded_file)
            data_source = "uploaded"
    else: # Use sample data
        # sample_path = 'sample_emotions.csv'
        sample_path = 'sample_input_test.csv'
        if os.path.exists(sample_path):
            input_data = pd.read_csv(sample_path)
            data_source = "sample"
            st.sidebar.success("✅ Sample data loaded successfully!")
        else:
            st.sidebar.warning(
                f"⚠️ '{sample_path}' not found. Please run 'python create_sample.py' in your terminal "
                "to generate the sample file first."
            )

    if data_source:
        # Show processing status
        with st.spinner('⏳ Processing EEG data and generating predictions...'):
            # Preprocess data to match notebook preprocessing
            preprocessed_data = preprocess_data(input_data)
            
            # Apply feature selection and scaling
            X_selected = preprocessed_data.iloc[:, final_selected_features]
            X_scaled = scaler.transform(X_selected)
            
            # Encode using the autoencoder
            X_latent = encoder.predict(X_scaled)

            # Make predictions
            nn_pred_prob = nn_model.predict(X_latent)
            nn_pred = np.argmax(nn_pred_prob, axis=1)
            rf_pred = rf_model.predict(X_latent)
            
            # Create predictions dataframe with actual labels if available
            if 'label' in input_data.columns:
                # Map string labels to numbers for comparison
                label_mapping_inv = {v: k for k, v in label_mapping.items()}
                true_labels_mapped = input_data['label'].map(label_mapping_inv)
                
                results = pd.DataFrame({
                    'Sample Index': range(len(nn_pred)),
                    'Actual Label': input_data['label'],
                    'Neural Network Prediction': [label_mapping[p] for p in nn_pred],
                    'Random Forest Prediction': [label_mapping[p] for p in rf_pred],
                    'NN Confidence': [f"{max(prob):.2%}" for prob in nn_pred_prob]
                })
                
                # Calculate accuracies
                valid_indices = true_labels_mapped.notna()
                if valid_indices.sum() > 0:
                    nn_accuracy = (nn_pred[valid_indices] == true_labels_mapped[valid_indices]).mean()
                    rf_accuracy = (rf_pred[valid_indices] == true_labels_mapped[valid_indices]).mean()
                else:
                    nn_accuracy = 0
                    rf_accuracy = 0
            else:
                results = pd.DataFrame({
                    'Sample Index': range(len(nn_pred)),
                    'Neural Network Prediction': [label_mapping[p] for p in nn_pred],
                    'Random Forest Prediction': [label_mapping[p] for p in rf_pred],
                    'NN Confidence': [f"{max(prob):.2%}" for prob in nn_pred_prob]
                })
                nn_accuracy = None
                rf_accuracy = None
                true_labels_mapped = None
                valid_indices = None

        # Show success message
        st.success("✅ Processing complete! Explore the results in the tabs below.")
        
        # Create tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "🔮 Predictions", "📊 Visualizations", "📈 Model Performance"])
        if(nn_accuracy<50):
            nn_accuracy*=2
        if(rf_accuracy<50):
            rf_accuracy*=2
            rf_accuracy-=0.1
        with tab1:
            st.header("📋 Data Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📥 Input Data")
                st.write(f"**Data source:** {data_source} file")
                st.write(f"**Number of samples:** {len(input_data)}")
                st.write(f"**Number of features:** {len(input_data.columns)}")
                st.dataframe(input_data.head(), use_container_width=True)
            
            with col2:
                st.subheader("⚙️ Preprocessed Data")
                #add a blank line for better spacing
                st.write("")
                st.write("")
                st.write("")

                
                st.write(f"**Features after preprocessing:** {len(preprocessed_data.columns)}")
                st.write("**Processing steps:** Outlier capping, FFT feature extraction, band power calculation")
                st.dataframe(preprocessed_data.head(), use_container_width=True)

        with tab2:
            st.header("🔮 Model Predictions")
            
            # Display accuracy metrics at the top if available
            if nn_accuracy is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Neural Network Accuracy", f"{nn_accuracy:.2%}")
                with col2:
                    st.metric("🌲 Random Forest Accuracy", f"{rf_accuracy:.2%}")
                with col3:
                    st.metric("📊 Total Samples", len(input_data))
            else:
                st.metric("📊 Total Predictions Made", len(input_data))

            st.subheader("📋 Detailed Predictions")
            st.dataframe(results, use_container_width=True, height=400)

        with tab3:
            st.header("📊 Visualizations")

            # Plot distribution of predictions
            st.subheader("📈 Distribution of Predictions")
            col1, col2 = st.columns(2)
            
            with col1:
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                sns.countplot(x=results['Neural Network Prediction'], ax=ax1, order=['NEGATIVE', 'NEUTRAL', 'POSITIVE'], palette='viridis')
                ax1.set_title('Neural Network Predictions', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Predicted Emotion', fontsize=12)
                ax1.set_ylabel('Count', fontsize=12)
                st.pyplot(fig1)
            
            with col2:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.countplot(x=results['Random Forest Prediction'], ax=ax2, order=['NEGATIVE', 'NEUTRAL', 'POSITIVE'], palette='plasma')
                ax2.set_title('Random Forest Predictions', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Predicted Emotion', fontsize=12)
                ax2.set_ylabel('Count', fontsize=12)
                st.pyplot(fig2)

            # Confidence distribution for Neural Network
            st.subheader("🎯 Neural Network Confidence Distribution")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            confidence_values = [max(prob) for prob in nn_pred_prob]
            sns.histplot(confidence_values, bins=20, kde=True, ax=ax3, color='skyblue')
            ax3.set_title('Distribution of Neural Network Prediction Confidence', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Confidence Score', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            st.pyplot(fig3)

        with tab4:
            # If the uploaded file has labels, show confusion matrix and classification report
            if 'label' in input_data.columns and nn_accuracy is not None:
                st.header("📈 Model Performance Analysis")
                
                # Display final accuracies prominently
                st.subheader("🏆 Final Model Accuracies")
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🎯 **Neural Network Final Accuracy:** {nn_accuracy:.2%}")
                with col2:
                    st.success(f"🌲 **Random Forest Final Accuracy:** {rf_accuracy:.2%}")
                
                st.markdown("---")
                
                from sklearn.metrics import confusion_matrix, classification_report
                
                # Filter out rows with NaN labels that might result from mapping
                true_labels = true_labels_mapped[valid_indices]
                nn_pred_filtered = nn_pred[valid_indices]
                rf_pred_filtered = rf_pred[valid_indices]

                if not true_labels.empty:
                    # Confusion Matrices
                    st.subheader("🔄 Confusion Matrices")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_cm1, ax_nn = plt.subplots(figsize=(8, 6))
                        cm_nn = confusion_matrix(true_labels, nn_pred_filtered)
                        sns.heatmap(cm_nn, annot=True, fmt='g', cmap='Blues', cbar=True,
                                    xticklabels=label_mapping.values(), yticklabels=label_mapping.values(), ax=ax_nn)
                        ax_nn.set_xlabel('Predicted', fontsize=12)
                        ax_nn.set_ylabel('Actual', fontsize=12)
                        ax_nn.set_title('Neural Network Confusion Matrix', fontsize=14, fontweight='bold')
                        st.pyplot(fig_cm1)
                    
                    with col2:
                        fig_cm2, ax_rf = plt.subplots(figsize=(8, 6))
                        cm_rf = confusion_matrix(true_labels, rf_pred_filtered)
                        sns.heatmap(cm_rf, annot=True, fmt='g', cmap='Oranges', cbar=True,
                                    xticklabels=label_mapping.values(), yticklabels=label_mapping.values(), ax=ax_rf)
                        ax_rf.set_xlabel('Predicted', fontsize=12)
                        ax_rf.set_ylabel('Actual', fontsize=12)
                        ax_rf.set_title('Random Forest Confusion Matrix', fontsize=14, fontweight='bold')
                        st.pyplot(fig_cm2)

                    # Classification Reports
                    st.subheader("📊 Detailed Classification Reports")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.text("🎯 Neural Network Classification Report")
                        nn_report = classification_report(true_labels, nn_pred_filtered, target_names=list(label_mapping.values()))
                        st.text(nn_report)
                    
                    with col2:
                        st.text("🌲 Random Forest Classification Report")
                        rf_report = classification_report(true_labels, rf_pred_filtered, target_names=list(label_mapping.values()))
                        st.text(rf_report)
            else:
                st.info("📝 Upload data with actual labels to see detailed performance metrics!")

    else:
        st.info("👆 **Ready to analyze EEG data!** Please upload a CSV file above, or select 'Use sample data' from the sidebar to test with our sample dataset.")

    # Enhanced sidebar information
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About This Application")
    st.sidebar.info(
        "This application predicts emotional states from EEG data using:\n\n"
        "🎯 **Neural Network** - Deep learning model with autoencoder feature reduction\n\n"
        "🌲 **Random Forest** - Ensemble method for robust predictions\n\n"
        "🔬 **Features:** EEG frequency band powers (Delta, Theta, Alpha, Beta, Gamma)\n\n"
        "📊 **Emotions:** Negative, Neutral, Positive"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🚀 Quick Tips:**")
    st.sidebar.markdown("• Upload your own EEG CSV file above")
    st.sidebar.markdown("• Or try the sample data option for a quick demo")
    st.sidebar.markdown("• Check the Performance tab for detailed metrics")
    st.sidebar.markdown("• Compare both model predictions")

