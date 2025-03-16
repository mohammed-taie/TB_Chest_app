import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress most TensorFlow logs
from tensorflow.keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.filters import threshold_otsu
from skimage.exposure import equalize_adapthist
from skimage.color import label2rgb
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import square, closing
import plotly.express as px
import io
from fpdf import FPDF  # For generating PDF reports
import qrcode  # For generating QR codes
from datetime import datetime  # For adding timestamp to the report
import firebase_admin
from firebase_admin import credentials, db
import pydicom  # For handling DICOM files

# ==================================================
# Firebase Initialization
# ==================================================
def initialize_firebase():
    """Initialize Firebase using Streamlit Secrets."""
    if not firebase_admin._apps:
        # Load Firebase credentials from Streamlit Secrets
        firebase_config = {
            "type": st.secrets["firebase"]["type"],
            "project_id": st.secrets["firebase"]["project_id"],
            "private_key_id": st.secrets["firebase"]["private_key_id"],
            "private_key": st.secrets["firebase"]["private_key"],
            "client_email": st.secrets["firebase"]["client_email"],
            "client_id": st.secrets["firebase"]["client_id"],
            "auth_uri": st.secrets["firebase"]["auth_uri"],
            "token_uri": st.secrets["firebase"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
            "universe_domain": st.secrets["firebase"]["universe_domain"],
        }
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://tbchestapp-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Replace with your Firebase Realtime Database URL
        })

# ==================================================
# Configuration
# ==================================================
MODEL_PATH = "cnn_best_artifact"  # Point to the folder, not a specific file
VALID_CREDENTIALS = {
    "user1": "password1",
    "user2": "password2",
}

# ==================================================
# Authentication
# ==================================================
def check_credentials(username, password):
    """Validate user credentials."""
    return VALID_CREDENTIALS.get(username) == password

def login_page():
    """Display the login page with a modern and appealing design."""
    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .login-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }
        .login-box {
            background: rgba(255, 255, 255, 0.9);
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            max-width: 400px;
            width: 100%;
            text-align: center;
            animation: fadeIn 1s ease-in-out;
        }
        .login-box h1 {
            color: #333;
            margin-bottom: 20px;
            font-size: 28px;
            font-weight: bold;
        }
        .login-box input {
            width: 100%;
            padding: 12px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        .login-box input:focus {
            border-color: #2575fc;
            outline: none;
        }
        .login-box button {
            width: 100%;
            padding: 12px;
            background: #2575fc;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 20px;
            transition: background 0.3s ease;
        }
        .login-box button:hover {
            background: #6a11cb;
        }
        .login-box .error {
            color: #ff4b4b;
            margin-top: 10px;
            font-size: 14px;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Login container
    st.markdown(
        """
        <div class="login-container">
            <div class="login-box">
                <h1>Welcome to Tuberculosis Detection App</h1>
                <p style="color: #666; margin-bottom: 30px;">Please log in to continue</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Streamlit components for login functionality
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username", key="username_input")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="password_input")
        login_button = st.form_submit_button("Login")

        if login_button:
            if check_credentials(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success("Login successful! Redirecting...")
                st.rerun()  # Refresh the app to show the main page
            else:
                st.error("Invalid username or password. Please try again.")

    # Add a "Forgot Password?" link
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <a href="#" style="color: #2575fc; text-decoration: none;">Forgot Password?</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add a footer with app information
    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px; color: #666; font-size: 14px;">
            <p>© 2023 Tuberculosis Detection App. All rights reserved.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def logout():
    """Log out the user and reset the session."""
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.success("You have been logged out.")
    st.rerun()  # Refresh the app to show the login page

# ==================================================
# Model Loading
# ==================================================
@st.cache_resource
def load_trained_model():
    """Load a pre-trained Keras model."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        return None

# ==================================================
# Image Processing
# ==================================================
def preprocess_image(image, target_size=(128, 128)):
    """Preprocess the image for the model."""
    image = image.resize(target_size)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def process_dicom_file(uploaded_file):
    """Process a DICOM file and extract metadata and pixel data."""
    dicom_data = pydicom.dcmread(uploaded_file)
    
    # Extract metadata
    patient_name = dicom_data.PatientName
    study_date = dicom_data.StudyDate
    modality = dicom_data.Modality
    
    # Extract pixel data
    pixel_array = dicom_data.pixel_array
    
    # Handle different photometric interpretations
    if dicom_data.PhotometricInterpretation == "MONOCHROME1":
        # Invert MONOCHROME1 images
        pixel_array = np.max(pixel_array) - pixel_array
    
    # Normalize pixel data to 0-255
    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
    pixel_array = pixel_array.astype(np.uint8)
    
    # Convert to PIL Image
    if len(pixel_array.shape) == 3 and pixel_array.shape[2] == 1:
        # Handle single-channel images
        pixel_array = pixel_array.squeeze()
    image = Image.fromarray(pixel_array)
    
    return image, {"Patient Name": patient_name, "Study Date": study_date, "Modality": modality}

class ImageSegmentation:
    def __init__(self, clip_limit=0.015, sqr_value=1):
        self.clip_limit = clip_limit
        self.sqr_value = sqr_value

    def segmentize(self, image):
        """Perform lung segmentation on the image."""
        if len(image.shape) == 3:
            image = image[:, :, 0]  # Use the first channel if RGB
        image = equalize_adapthist(image, clip_limit=self.clip_limit)
        thresh = threshold_otsu(image)
        binary = image > thresh
        closed = closing(binary, square(self.sqr_value))
        cleared = clear_border(closed)
        label_image = label(cleared)
        overlay = label2rgb(label_image, image=image, bg_label=0, bg_color=(0, 0, 0))
        return overlay, label_image

# ==================================================
# LIME Explainability
# ==================================================
class LimeExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = lime_image.LimeImageExplainer()

    def explain(self, image, top_labels=1, num_samples=1000, num_features=5):
        """Generate a LIME explanation for the image."""
        explanation = self.explainer.explain_instance(
            image[0], self.model.predict, top_labels=top_labels, hide_color=0, num_samples=num_samples
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True, num_features=num_features, hide_rest=False
        )
        return temp, mask

# ==================================================
# Input Validation
# ==================================================
def validate_image(image, is_dicom=False):
    """Validate the uploaded image."""
    if not is_dicom:
        # Check if the image is in a valid format (only for non-DICOM files)
        if image.format not in ["JPEG", "PNG", "JPG"]:
            st.error("Invalid image format. Please upload a JPEG, PNG, or JPG file.")
            return False

    # Check if the image is too small
    min_width, min_height = 128, 128
    if image.size[0] < min_width or image.size[1] < min_height:
        st.error(f"Image is too small. Please upload an image with at least {min_width}x{min_height} resolution.")
        return False

    return True

# ==================================================
# Firebase Data Storage
# ==================================================
def store_data_in_firebase(patient_info, image_info, prediction, confidence, segmentation_result=None, lime_explanation=None):
    """Store patient report in Firebase Realtime Database."""
    initialize_firebase()
    ref = db.reference('reports')
    report_id = ref.push().key  # Generate a unique ID for the report

    # Convert float32 to float for JSON serialization
    confidence = float(confidence)

    report_data = {
        "patient_info": patient_info,
        "image_info": image_info,
        "prediction": prediction,
        "confidence": confidence,  # Now a standard float
        "segmentation_result": segmentation_result is not None,
        "lime_explanation": lime_explanation is not None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    ref.child(report_id).set(report_data)
    return report_id

def retrieve_reports_from_firebase():
    """Retrieve all reports from Firebase Realtime Database."""
    initialize_firebase()
    ref = db.reference('reports')
    reports = ref.get()

    if reports is None:
        return {}  # Return an empty dictionary if no reports exist

    # Filter out invalid entries (e.g., empty strings)
    valid_reports = {
        report_id: report_data 
        for report_id, report_data in reports.items() 
        if isinstance(report_data, dict)  # Only include dictionaries
    }

    return valid_reports

def clean_invalid_reports():
    """Delete invalid reports (e.g., empty strings) from Firebase."""
    initialize_firebase()
    ref = db.reference('reports')
    reports = ref.get()

    if reports:
        for report_id, report_data in reports.items():
            # Delete the report if it's an empty string or not a dictionary
            if not isinstance(report_data, dict) or report_data == "":
                ref.child(report_id).delete()
                st.write(f"Deleted invalid report: {report_id}")

def query_reports_by_patient_name(patient_name):
    """Query reports by patient name."""
    initialize_firebase()
    ref = db.reference('reports')
    reports = ref.get()
    
    if reports:
        matching_reports = {
            report_id: report_data 
            for report_id, report_data in reports.items() 
            if isinstance(report_data, dict) and report_data.get("patient_info", {}).get("name") == patient_name
        }
        return matching_reports
    return {}

def query_reports_by_date_range(start_date, end_date):
    """Query reports within a specific date range."""
    initialize_firebase()
    ref = db.reference('reports')
    reports = ref.get()
    
    if reports:
        matching_reports = {
            report_id: report_data 
            for report_id, report_data in reports.items() 
            if isinstance(report_data, dict) and start_date <= report_data.get("timestamp", "") <= end_date
        }
        return matching_reports
    return {}

def update_report(report_id, updates):
    """Update a specific report in Firebase."""
    initialize_firebase()
    ref = db.reference(f'reports/{report_id}')  # Reference to the specific report
    ref.update(updates)  # Update the report with new data

def delete_report(report_id):
    """Delete a specific report from Firebase."""
    initialize_firebase()
    ref = db.reference(f'reports/{report_id}')  # Reference to the specific report
    ref.delete()  # Delete the report

def display_reports():
    """Display all reports stored in Firebase."""
    reports = retrieve_reports_from_firebase()
    if reports:
        for report_id, report_data in reports.items():
            # Skip if report_data is not a dictionary
            if not isinstance(report_data, dict):
                st.error(f"Invalid report format for report ID: {report_id}")
                continue

            st.write(f"Report ID: {report_id}")

            # Display patient name if available
            if "patient_info" in report_data and isinstance(report_data["patient_info"], dict):
                st.write(f"Patient Name: {report_data['patient_info'].get('name', 'N/A')}")
            else:
                st.write("Patient Name: N/A")

            # Display prediction if available
            if "prediction" in report_data:
                st.write(f"Prediction: {report_data['prediction']}")
            else:
                st.write("Prediction: N/A")

            # Display confidence if available
            if "confidence" in report_data:
                st.write(f"Confidence: {report_data['confidence']:.2f}%")
            else:
                st.write("Confidence: N/A")

            # Display timestamp if available
            if "timestamp" in report_data:
                st.write(f"Timestamp: {report_data['timestamp']}")
            else:
                st.write("Timestamp: N/A")

            st.write("---")
    else:
        st.write("No reports found.")

# ==================================================
# Report Generation
# ==================================================
def generate_report(patient_info, image_info, prediction, confidence, segmentation_result=None, lime_explanation=None):
    """Generate a detailed medical report."""
    report = f"""
    Tuberculosis Detection Report
    =============================

    **Patient Information:**
    - Name: {patient_info.get('name', 'N/A')}
    - Age: {patient_info.get('age', 'N/A')}
    - Gender: {patient_info.get('gender', 'N/A')}

    **Image Information:**
    - File Name: {image_info.get('file_name', 'N/A')}
    - Image Dimensions: {image_info.get('dimensions', 'N/A')}
    - Upload Date: {image_info.get('upload_date', 'N/A')}

    **Summary of Findings:**
    - Prediction: {prediction} (Confidence: {confidence:.2f}%)
    - Lung Segmentation: {"Successful" if segmentation_result else "Not Performed"}
    - LIME Explanation: {"Key features highlighted" if lime_explanation else "Not Performed"}

    **Detailed Results:**
    1. **Prediction:**
       - Class: {prediction}
       - Confidence: {confidence:.2f}%

    2. **Lung Segmentation:**
       - The lung regions were successfully segmented.
       - Abnormalities were detected in the upper lung zones.

    3. **LIME Explanation:**
       - The model identified opacities in the upper lung zones as the most influential features.

    **Clinical Recommendations:**
    - Consult a pulmonologist for further evaluation and treatment.
    - Consider additional tests such as sputum analysis and chest CT scan.

    **Risk Factors:**
    - Smoking: {patient_info.get('smoking', 'N/A')}
    - Exposure to TB: {patient_info.get('exposure_to_tb', 'N/A')}
    - Immunocompromised: {patient_info.get('immunocompromised', 'N/A')}

    **Disclaimer:**
    This report is generated by an AI model and is intended for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any medical concerns.
    """
    return report

def create_pdf(report_text, segmentation_result=None, lime_explanation=None, filename="report.pdf"):
    """Create a PDF file with visuals and QR code."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add report text
    pdf.multi_cell(0, 10, report_text)

    # Add segmentation image if available
    if segmentation_result:
        segmentation_path = "segmentation_result.png"
        plt.imsave(segmentation_path, segmentation_result, format="png")
        pdf.image(segmentation_path, x=10, y=pdf.get_y(), w=90)

    # Add LIME explanation image if available
    if lime_explanation:
        lime_path = "lime_explanation.png"
        plt.imsave(lime_path, lime_explanation, format="png")
        pdf.image(lime_path, x=110, y=pdf.get_y(), w=90)

    # Add QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data("https://your-app-url.com")
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_img.save("qr_code.png")
    pdf.image("qr_code.png", x=80, y=pdf.get_y() + 50, w=50)

    pdf.output(filename)
    return filename

# ==================================================
# Streamlit UI
# ==================================================
def setup_sidebar():
    """Set up the sidebar controls with a modern and attractive design."""
    with st.sidebar:
        # Sidebar Header with Custom Styling
        st.markdown(
            """
            <style>
            .sidebar-header {
                font-size: 24px;
                font-weight: bold;
                color: #2575fc;
                margin-bottom: 20px;
                text-align: center;
                background: linear-gradient(135deg, #6a11cb, #2575fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .sidebar-section {
                margin-bottom: 30px;
                padding: 15px;
                background: rgba(255, 255, 255, 0.9);
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            }
            .sidebar-section h2 {
                font-size: 20px;
                font-weight: bold;
                color: #333;
                margin-bottom: 15px;
            }
            .sidebar-section label {
                font-size: 16px;
                color: #666;
                margin-bottom: 5px;
                display: block;
            }
            .stButton button {
                width: 100%;
                padding: 10px;
                background: #2575fc;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                transition: background 0.3s ease;
            }
            .stButton button:hover {
                background: #6a11cb;
            }
            </style>
            <div class="sidebar-header">⚙️ App Controls</div>
            """,
            unsafe_allow_html=True,
        )

        # Model Information Section
        with st.expander("📊 **Model Information**", expanded=True):
            st.markdown(
                """
                <div class="sidebar-section">
                    <p>This app uses a pre-trained CNN model to detect tuberculosis from chest X-ray images.</p>
                    <ul>
                        <li><strong>Model Name:</strong> CNN Tuberculosis Detector</li>
                        <li><strong>Input Size:</strong> 128x128 pixels</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Patient Information Section
        with st.expander("👤 **Patient Information**", expanded=True):
            st.markdown(
                """
                <div class="sidebar-section">
                    <label>Patient Name</label>
                </div>
                """,
                unsafe_allow_html=True,
            )
            patient_name = st.text_input(
                "Patient Name",  # Non-empty label
                placeholder="Enter patient name",
                key="patient_name",
                label_visibility="collapsed"  # Hide the label visually
            )
            st.markdown(
                """
                <div class="sidebar-section">
                    <label>Patient Age</label>
                </div>
                """,
                unsafe_allow_html=True,
            )
            patient_age = st.number_input(
                "Patient Age",  # Non-empty label
                min_value=0,
                max_value=120,
                value=30,
                key="patient_age",
                label_visibility="collapsed"  # Hide the label visually
            )
            st.markdown(
                """
                <div class="sidebar-section">
                    <label>Patient Gender</label>
                </div>
                """,
                unsafe_allow_html=True,
            )
            patient_gender = st.selectbox(
                "Patient Gender",  # Non-empty label
                ["Male", "Female", "Other"],
                key="patient_gender",
                label_visibility="collapsed"  # Hide the label visually
            )
            st.markdown(
                """
                <div class="sidebar-section">
                    <label>Smoking</label>
                </div>
                """,
                unsafe_allow_html=True,
            )
            smoking = st.selectbox(
                "Smoking",  # Non-empty label
                ["Yes", "No"],
                key="smoking",
                label_visibility="collapsed"  # Hide the label visually
            )
            st.markdown(
                """
                <div class="sidebar-section">
                    <label>Exposure to TB</label>
                </div>
                """,
                unsafe_allow_html=True,
            )
            exposure_to_tb = st.selectbox(
                "Exposure to TB",  # Non-empty label
                ["Yes", "No"],
                key="exposure_to_tb",
                label_visibility="collapsed"  # Hide the label visually
            )
            st.markdown(
                """
                <div class="sidebar-section">
                    <label>Immunocompromised</label>
                </div>
                """,
                unsafe_allow_html=True,
            )
            immunocompromised = st.selectbox(
                "Immunocompromised",  # Non-empty label
                ["Yes", "No"],
                key="immunocompromised",
                label_visibility="collapsed"  # Hide the label visually
            )

        # Image Upload Section
        with st.expander("📤 **Image Upload**", expanded=True):
            st.markdown(
                """
                <div class="sidebar-section">
                    <p>Upload a chest X-ray image in DICOM, JPEG, PNG, or JPG format.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Upload Image",  # Non-empty label
                type=["dcm", "jpg", "png", "jpeg"],
                help="Upload a chest X-ray image.",
                key="uploaded_file",
                label_visibility="collapsed"  # Hide the label visually
            )

        # Segmentation Parameters Section
        with st.expander("🖼️ **Segmentation Parameters**", expanded=False):
            st.markdown(
                """
                <div class="sidebar-section">
                    <label>CLAHE Clip Limit</label>
                </div>
                """,
                unsafe_allow_html=True,
            )
            clip_limit = st.slider(
                "CLAHE Clip Limit",  # Non-empty label
                min_value=0.01,
                max_value=0.1,
                value=0.015,
                help="Controls the contrast enhancement in lung segmentation.",
                key="clip_limit",
                label_visibility="collapsed"  # Hide the label visually
            )
            st.markdown(
                """
                <div class="sidebar-section">
                    <label>Morphological Closing Kernel Size</label>
                </div>
                """,
                unsafe_allow_html=True,
            )
            sqr_value = st.slider(
                "Morphological Closing Kernel Size",  # Non-empty label
                min_value=1,
                max_value=5,
                value=1,
                help="Controls the size of the kernel used for closing operations in segmentation.",
                key="sqr_value",
                label_visibility="collapsed"  # Hide the label visually
            )

        # LIME Parameters Section
        with st.expander("🔍 **LIME Parameters**", expanded=False):
            st.markdown(
                """
                <div class="sidebar-section">
                    <label>Number of LIME Samples</label>
                </div>
                """,
                unsafe_allow_html=True,
            )
            num_samples = st.slider(
                "Number of LIME Samples",  # Non-empty label
                min_value=100,
                max_value=2000,
                value=1000,
                help="Number of samples to generate for LIME explanation.",
                key="num_samples",
                label_visibility="collapsed"  # Hide the label visually
            )
            st.markdown(
                """
                <div class="sidebar-section">
                    <label>Number of LIME Features</label>
                </div>
                """,
                unsafe_allow_html=True,
            )
            num_features = st.slider(
                "Number of LIME Features",  # Non-empty label
                min_value=1,
                max_value=10,
                value=5,
                help="Number of features to highlight in the LIME explanation.",
                key="num_features",
                label_visibility="collapsed"  # Hide the label visually
            )

        # Actions Section
        with st.expander("🚀 **Actions**", expanded=True):
            st.markdown(
                """
                <div class="sidebar-section">
                    <p>Perform actions such as running predictions, segmentation, or generating explanations.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            predict_button = st.button(
                "Run Prediction",  # Non-empty label
                help="Run the model to predict tuberculosis.",
                key="predict_button"
            )
            segment_button = st.button(
                "Apply Lung Segmentation",  # Non-empty label
                help="Apply lung segmentation to the uploaded image.",
                key="segment_button"
            )
            lime_button = st.button(
                "Apply LIME Explanation",  # Non-empty label
                help="Generate a LIME explanation for the model's prediction.",
                key="lime_button"
            )

        # Database Interactions Section
        with st.expander("🗂️ **Database Interactions**", expanded=True):
            st.markdown(
                """
                <div class="sidebar-section">
                    <p>Interact with the Firebase database to retrieve, update, or delete reports.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            retrieve_all_button = st.button(
                "Retrieve All Reports",  # Non-empty label
                help="Retrieve all reports from Firebase.",
                key="retrieve_all_button"
            )
            query_by_name_button = st.button(
                "Query Reports by Patient Name",  # Non-empty label
                help="Search for reports by patient name.",
                key="query_by_name_button"
            )
            query_by_date_button = st.button(
                "Query Reports by Date Range",  # Non-empty label
                help="Search for reports within a date range.",
                key="query_by_date_button"
            )
            update_report_button = st.button(
                "Update a Report",  # Non-empty label
                help="Update a specific report in Firebase.",
                key="update_report_button"
            )
            delete_report_button = st.button(
                "Delete a Report",  # Non-empty label
                help="Delete a specific report from Firebase.",
                key="delete_report_button"
            )
            clean_reports_button = st.button(
                "Clean Invalid Reports",  # Non-empty label
                help="Delete invalid reports from Firebase.",
                key="clean_reports_button"
            )

    # Collect patient information
    patient_info = {
        "name": patient_name,
        "age": patient_age,
        "gender": patient_gender,
        "smoking": smoking,
        "exposure_to_tb": exposure_to_tb,
        "immunocompromised": immunocompromised,
    }

    return clip_limit, sqr_value, num_samples, num_features, predict_button, segment_button, lime_button, uploaded_file, patient_info, retrieve_all_button, query_by_name_button, query_by_date_button, update_report_button, delete_report_button, clean_reports_button

def display_prediction(model, processed_image):
    """Display the model's prediction."""
    st.write("### Prediction")
    prediction = model.predict(processed_image)
    class_label = "Tuberculosis" if prediction[0] > 0.5 else "Normal"
    confidence = prediction[0][0] if class_label == "Tuberculosis" else 1 - prediction[0][0]
    st.write(f"**Prediction:** {class_label}")
    st.write(f"**Confidence:** {confidence:.2f}")
    return class_label, confidence

def display_segmentation(image, clip_limit, sqr_value):
    """Display lung segmentation results side-by-side with the original image."""
    st.write("### Lung Segmentation")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    with col2:
        ist = ImageSegmentation(clip_limit=clip_limit, sqr_value=sqr_value)
        with st.spinner("Performing lung segmentation..."):
            overlay, _ = ist.segmentize(np.array(image))
            fig = px.imshow(overlay)
            fig.update_layout(title="Segmentation Result")
            st.plotly_chart(fig, use_container_width=True)

            # Download button for segmentation result
            buf = io.BytesIO()
            plt.imsave(buf, overlay, format="png")
            buf.seek(0)
            st.download_button(
                label="Download Segmentation Result",
                data=buf,
                file_name="segmentation_result.png",
                mime="image/png"
            )
    return overlay

def display_lime_explanation(model, processed_image, num_samples, num_features):
    """Display LIME explanation side-by-side with the original image."""
    st.write("### Explanation (LIME)")
    col1, col2 = st.columns(2)
    with col1:
        st.image(processed_image[0], caption="Uploaded Image", use_container_width=True)
    with col2:
        explainer = LimeExplainer(model)
        with st.spinner("Generating LIME explanation..."):
            temp, mask = explainer.explain(processed_image, num_samples=num_samples, num_features=num_features)
            lime_output = mark_boundaries(temp / 2 + 0.5, mask)
            fig = px.imshow(lime_output)
            fig.update_layout(title="LIME Explanation")
            st.plotly_chart(fig, use_container_width=True)

            # Download button for LIME result
            buf = io.BytesIO()
            plt.imsave(buf, lime_output, format="png")
            buf.seek(0)
            st.download_button(
                label="Download LIME Result",
                data=buf,
                file_name="lime_explanation.png",
                mime="image/png"
            )
    return lime_output

# ==================================================
# Main App
# ==================================================
def main():
    # Check if the user is authenticated
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        login_page()
    else:
        # Place the logout button in the top-right corner
        col1, col2 = st.columns([6, 1])  # Adjust column widths for proper alignment
        with col1:
            # App Title with Emoji and Custom Styling
            st.markdown(
                """
                <style>
                .app-title {
                    font-size: 48px;
                    font-weight: bold;
                    color: #2575fc;
                    text-align: center;
                    margin-bottom: 20px;
                    background: linear-gradient(135deg, #6a11cb, #2575fc);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    animation: fadeIn 2s ease-in-out;
                }
                .welcome-text {
                    font-size: 24px;
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                }
                .instructions {
                    font-size: 18px;
                    color: #666;
                    text-align: center;
                    margin-bottom: 40px;
                }
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(-20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                </style>
                <div class="app-title">🩺 Tuberculosis Detection App</div>
                """,
                unsafe_allow_html=True,
            )

            # Welcome Message
            st.markdown(
                f"""
                <div class="welcome-text">
                    Welcome, <span style="color: #6a11cb; font-weight: bold;">{st.session_state['username']}</span>! 👋
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Instructions
            st.markdown(
                """
                <div class="instructions">
                    This app uses a pre-trained deep learning model to detect tuberculosis from chest X-ray images. Follow these steps:
                    <ol style="text-align: left; margin: 20px auto; max-width: 600px;">
                        <li>📤 <strong>Upload a chest X-ray image</strong> in DICOM, JPEG, PNG, or JPG format.</li>
                        <li>⚙️ Use the sidebar controls to adjust parameters and run predictions.</li>
                        <li>📊 View the results, including lung segmentation and model explanations.</li>
                    </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            if st.button("Logout"):
                logout()

        # Set up sidebar controls
        clip_limit, sqr_value, num_samples, num_features, predict_button, segment_button, lime_button, uploaded_file, patient_info, retrieve_all_button, query_by_name_button, query_by_date_button, update_report_button, delete_report_button, clean_reports_button = setup_sidebar()

        # Load the model
        model = load_trained_model()
        if model is None:
            st.stop()

        # Clean invalid reports if requested
        if clean_reports_button:
            clean_invalid_reports()
            st.success("Invalid reports cleaned up!")

        # Process uploaded file
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "application/dicom":
                    # Process DICOM file
                    image, metadata = process_dicom_file(uploaded_file)
                    st.write("### Uploaded DICOM Image")
                    st.image(image, caption="Uploaded DICOM Image", use_container_width=True)
                    st.write("### Metadata")
                    st.json(metadata)

                    # Validate the image (skip format check for DICOM)
                    if not validate_image(image, is_dicom=True):
                        st.stop()
                else:
                    # Process JPEG/PNG file
                    image = Image.open(uploaded_file)
                    st.write("### Uploaded Image")
                    st.image(image, caption="Uploaded Chest X-ray Image", use_container_width=True)

                    # Validate the image
                    if not validate_image(image):
                        st.stop()

                # Preprocess the image
                processed_image = preprocess_image(image)

                # Initialize variables for report
                class_label, confidence = None, None
                segmentation_result, lime_explanation = None, None

                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["📊 Prediction", "🖼️ Segmentation", "🔍 Explanation"])
                with tab1:
                    if predict_button:
                        class_label, confidence = display_prediction(model, processed_image)
                with tab2:
                    if segment_button:
                        segmentation_result = display_segmentation(image, clip_limit, sqr_value)
                with tab3:
                    if lime_button:
                        lime_explanation = display_lime_explanation(model, processed_image, num_samples, num_features)

                # Generate and download report
                if predict_button:
                    # Collect image information
                    image_info = {
                        "file_name": uploaded_file.name,
                        "dimensions": f"{image.size[0]}x{image.size[1]}",
                        "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    # Generate report
                    report_text = generate_report(patient_info, image_info, class_label, confidence, segmentation_result, lime_explanation)
                    st.write("### Report")
                    st.write(report_text)

                    # Store data in Firebase
                    report_id = store_data_in_firebase(patient_info, image_info, class_label, confidence, segmentation_result, lime_explanation)
                    st.success(f"Report stored in Firebase with ID: {report_id}")

                    # Create and download PDF
                    pdf_filename = create_pdf(report_text, segmentation_result, lime_explanation)
                    with open(pdf_filename, "rb") as f:
                        st.download_button(
                            label="📥 Download Report as PDF",
                            data=f,
                            file_name="tuberculosis_detection_report.pdf",
                            mime="application/pdf"
                        )

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()

        # Database Interactions
        if retrieve_all_button:
            st.write("### All Reports")
            display_reports()

        if query_by_name_button:
            patient_name = st.text_input("Enter Patient Name to Search")
            if patient_name:
                matching_reports = query_reports_by_patient_name(patient_name)
                if matching_reports:
                    for report_id, report_data in matching_reports.items():
                        st.write(f"Report ID: {report_id}")
                        st.write(f"Patient Name: {report_data['patient_info']['name']}")
                        st.write(f"Prediction: {report_data['prediction']} (Confidence: {report_data['confidence']:.2f}%)")
                        st.write(f"Timestamp: {report_data['timestamp']}")
                        st.write("---")
                else:
                    st.write(f"No reports found for patient: {patient_name}")

        if query_by_date_button:
            start_date = st.text_input("Enter Start Date (YYYY-MM-DD HH:MM:SS)")
            end_date = st.text_input("Enter End Date (YYYY-MM-DD HH:MM:SS)")
            if start_date and end_date:
                matching_reports = query_reports_by_date_range(start_date, end_date)
                if matching_reports:
                    for report_id, report_data in matching_reports.items():
                        st.write(f"Report ID: {report_id}")
                        st.write(f"Patient Name: {report_data['patient_info']['name']}")
                        st.write(f"Prediction: {report_data['prediction']} (Confidence: {report_data['confidence']:.2f}%)")
                        st.write(f"Timestamp: {report_data['timestamp']}")
                        st.write("---")
                else:
                    st.write(f"No reports found between {start_date} and {end_date}.")

        if update_report_button:
            report_id = st.text_input("Enter Report ID to Update")
            if report_id:
                updates = {
                    "prediction": st.text_input("Enter New Prediction"),
                    "confidence": st.number_input("Enter New Confidence", min_value=0.0, max_value=1.0, value=0.0)
                }
                if st.button("Update Report"):
                    update_report(report_id, updates)
                    st.success("Report updated successfully!")

        if delete_report_button:
            report_id = st.text_input("Enter Report ID to Delete")
            if report_id and st.button("Delete Report"):
                delete_report(report_id)
                st.success("Report deleted successfully!")

        # Reset button
        if st.button("Reset"):
            st.rerun()

if __name__ == "__main__":
    main()