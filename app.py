import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from PIL import Image
import io
import shutil

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)  # Allow frontend to talk to backend


def get_model():
    """
    Tries to get the model from Flask's 'g' object.
    If it's not there, it trains and stores it.
    """
    if 'model' not in g:
        print("💡 Model not found in g. Training model...")
        g.model = train_model()
    return g.model

# ---------- STEP 1: Define dataset path ----------
#

#
DATASET_PATH = "dataset" 

# ---------- STEP 2: The Training Function (Eigenfaces Math) ----------
def train_model():
    """
    Loads all images from DATASET_PATH and computes the Eigenfaces.
    This is your original code, refactored into a function.
    """
    print(f"📂 Loading dataset from: {DATASET_PATH}")
    images = []
    labels = []
    label_names = []
    label_id = 0

    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: Dataset path not found: {DATASET_PATH}")
        # Create it if it doesn't exist, so the app doesn't crash
        os.makedirs(DATASET_PATH) 
        
    for person in sorted(os.listdir(DATASET_PATH)):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue

        label_names.append(person)
        for file in os.listdir(person_path):
            img_path = os.path.join(person_path, file)
            try:
                # Try with OpenCV
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Fallback for formats like .pgm
                if img is None:
                    img_pil = Image.open(img_path).convert('L')
                    img = np.array(img_pil, dtype=np.uint8)

                img_resized = cv2.resize(img, (100, 100))
                images.append(img_resized.flatten())
                labels.append(label_id)
            except Exception as e:
                print(f"⚠️ Error reading {img_path}: {e}")
        label_id += 1

    if not images:
        print("⚠️ No images found. Model is empty.")
        # Return empty-but-valid model components
        return {
            "X": np.array([]),
            "y": np.array([]),
            "label_names": [],
            "mean_face": np.zeros((10000, 1)),
            "eigenfaces": np.zeros((10000, 1)),
            "projected_train": np.array([])
        }

    X = np.array(images).T  # Shape: (10000, num_images)
    y = np.array(labels)
    
    print(f"✅ Loaded {X.shape[1]} images for {len(label_names)} people.")

    # Compute mean face
    mean_face = np.mean(X, axis=1).reshape(-1, 1)

    # Compute eigenfaces (PCA)
    A = X - mean_face
    
    # --- The "PCA Trick" ---
    # We compute L = A.T @ A (small m x m matrix)
    # instead of C = A @ A.T (huge N x N matrix)
    L = np.dot(A.T, A)
    eigenvalues, eigenvectors_L = np.linalg.eig(L)

    # Get the real eigenfaces (u_i = A @ v_i)
    eigenvectors_C = np.dot(A, eigenvectors_L)

    # Sort by importance (highest eigenvalue)
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors_C = eigenvectors_C[:, idx]

    # Normalize (Eigenvectors must be unit vectors)
    eigenfaces = eigenvectors_C / np.linalg.norm(eigenvectors_C, axis=0)

    N_COMPONENTS = 50  # Keep top 50 components
    eigenfaces = eigenfaces[:, :N_COMPONENTS]

    # Project all training images into the new "face space"
    projected_train = np.dot(eigenfaces.T, A)
    
    print("✅ Model training complete.")
    
    # Return all model components in a dictionary
    return {
        "X": X,
        "y": y,
        "label_names": label_names,
        "mean_face": mean_face,
        "eigenfaces": eigenfaces,
        "projected_train": projected_train
    }

# ---------- STEP 3: Helper function for image conversion ----------
def numpy_to_base64(img_array):
    """Converts a 100x100 numpy array to a base64 string"""
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- API ENDPOINT 1: Recognize a face ---
@app.route('/recognize', methods=['POST'])
def recognize_face_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    # Read image from upload
    try:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        # Fallback for tricky formats
        if img is None:
            file.seek(0) # Rewind file
            img_pil = Image.open(file).convert('L')
            img = np.array(img_pil, dtype=np.uint8)
            
        img_resized = cv2.resize(img, (100, 100))
        img_flat = img_resized.flatten().reshape(-1, 1)

    except Exception as e:
        return jsonify({'error': f'Could not read image: {e}'}), 400

    # Get the trained model
    model = get_model()
    if model["X"].shape[0] == 0:
         return jsonify({'error': 'Model is not trained. Add images first.'}), 500

    # --- Run recognition math ---
    diff = img_flat - model["mean_face"]
    projected_test = np.dot(model["eigenfaces"].T, diff)

    distances = np.linalg.norm(model["projected_train"] - projected_test, axis=0)
    min_index = np.argmin(distances)

    predicted_label = model["y"][min_index]
    predicted_person = model["label_names"][predicted_label]
    min_distance = distances[min_index]

    # Get matched image and convert to base64
    match_img_flat = model["X"][:, min_index]
    match_img_resized = match_img_flat.reshape(100, 100)
    
    # --- Send JSON response ---
    return jsonify({
        'person': predicted_person,
        'distance': f'{min_distance:.2f}',
        'test_image_b64': numpy_to_base64(img_resized),
        'match_image_b64': numpy_to_base64(match_img_resized)
    })

# --- API ENDPOINT 2: Add a new person ---
@app.route('/add_person', methods=['POST'])
def add_person_endpoint():
    person_name = request.form.get('personName')
    images = request.files.getlist('image')

    if not person_name:
        return jsonify({'error': 'No person name provided'}), 400
    if not images:
        return jsonify({'error': 'No images provided'}), 400

    # Create the directory for the new person (e.g., "dataset/John_Doe")
    # This is a basic way to sanitize; a real app would be more robust
    safe_person_name = "".join(c for c in person_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    person_dir = os.path.join(DATASET_PATH, safe_person_name)
    
    os.makedirs(person_dir, exist_ok=True)

    # Save all uploaded images
    for i, file in enumerate(images):
        file.save(os.path.join(person_dir, f'{i+1}.png')) # Save as png

    # --- CRITICAL: Retrain the model ---
    # Invalidate the old model
    if 'model' in g:
        g.pop('model', None) 
    
    # Train a new one. The next request will pick this up.
    g.model = train_model()
    
    return jsonify({
        'message': f'Successfully added {person_name} with {len(images)} images. Model has been retrained.'
    })

# --- API ENDPOINT 3: Get model visualization details ---
@app.route('/get_model_details', methods=['GET'])
def get_model_details_endpoint():
    model = get_model()
    
    if model["X"].shape[0] == 0:
         return jsonify({'error': 'Model is not trained. Add images first.'}), 500
         
    # Convert mean face
    mean_face_b64 = numpy_to_base64(model["mean_face"].reshape(100, 100))
    
    # Convert top 5 eigenfaces
    eigenfaces_b64 = []
    for i in range(5):
        # We need to scale the eigenface to 0-255 to be viewable
        ef = model["eigenfaces"][:, i].reshape(100, 100)
        ef_scaled = cv2.normalize(ef, None, 0, 255, cv2.NORM_MINMAX)
        eigenfaces_b64.append(numpy_to_base64(ef_scaled))
        
    return jsonify({
        'mean_face_b64': mean_face_b64,
        'eigenfaces_b64': eigenfaces_b64
    })

# --- Main entry point to run the server ---
if __name__ == '__main__':
    # We call train_model() once on startup to
    # "warm up" the server and create the first model.
    with app.app_context():
        g.model = train_model()
    
    print("\n🚀 Backend server is running!")
    print("You can now open your index.html file in a browser.")
    app.run(debug=True, port=5000)

