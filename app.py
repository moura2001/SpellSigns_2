import os
import cv2
from flask import Flask, render_template, request,send_file, send_from_directory,jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from transformers import MarianMTModel, MarianTokenizer
import nltk
from nltk.tokenize import word_tokenize
import string
from collections import namedtuple
import consts
from huggingface_hub import login, hf_hub_download
import tensorflow as tf

login("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")  # Replace with your actual token

arabic_model_path = hf_hub_download(repo_id="mourasaber2001/SpellSigns", filename="arabic_model.h5",use_auth_token=True
)
american_model_path = hf_hub_download(repo_id="mourasaber2001/SpellSigns", filename="english_model.h5",use_auth_token=True
)

arabic_model = load_model(arabic_model_path)
american_model = load_model(american_model_path)


arabic_encoder = joblib.load("./models/arabic_encoder.pkl")
american_encoder = joblib.load("./models/english_encoder.pkl")

# Dictionary mapping languages to their respective alphabets
alphabets = {
    'american': list('abcdefghijklmnopqrstuvwxyz'),
    'arabic': ['أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']
    # Add more languages and their alphabets here
}


UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def file_is_allowed(filename: str):
    return ('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def get_image_prediction(file, model, encoder) -> str:
    # Load the image
    nparr = np.frombuffer(file.read(), np.uint8)  # Use frombuffer instead of fromstring
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Could not open or find the image.")
    
    # Resize image to (224, 224)
    img_resized = cv2.resize(image, (224, 224))

    # Step 1: Grayscale conversion
    gray_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Step 2: Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)  # You can adjust the thresholds

    # Step 3: Prepare the image for prediction
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges back to 3 channels (RGB)
    image_batch = np.expand_dims(edges_rgb, axis=0)  # Add batch dimension (shape: (1, 224, 224, 3))

    # Perform prediction
    prediction = model.predict(image_batch)
    predicted_class_index = np.argmax(prediction, axis=1)  # Get the index of the class with the highest probability

    # One-hot encoding for inverse transformation
    predicted_class_one_hot = np.zeros((predicted_class_index.size, encoder.categories_[0].size))
    predicted_class_one_hot[np.arange(predicted_class_index.size), predicted_class_index] = 1
    predicted_class_label = encoder.inverse_transform(predicted_class_one_hot)[0][0]
    # Convert predicted_class_label to string if it's not already
    if isinstance(predicted_class_label, np.ndarray):
        predicted_class_label = predicted_class_label.item()  # Convert to a single string if it's an array

    return str(predicted_class_label)  # Ensure return types are str and bool



@app.route('/')
def index():
    return render_template("index.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)            

def get_model_for_language(language):
    # Replace with logic to return the appropriate model based on the language
    if language == "arabic":
        return arabic_model  # Your Arabic model
    elif language == "american":
        return american_model  # Your English model
    # Add more languages as necessary

def get_encoder_for_language(language):
    # Replace with logic to return the appropriate encoder based on the language
    if language == "arabic":
        return arabic_encoder  # Your Arabic encoder
    elif language == "american":
        return american_encoder  # Your English encoder
    # Add more languages as necessary


                
@app.route("/txt/<language>", methods=['GET', 'POST'])
def classification(language):
    # Get alphabet based on the language
    alphabet = alphabets.get(language, alphabets['american'])
    if language not in ['arabic', 'american']:
        return "Invalid language", 404

    if request.method == "POST":
        # Handle the selected letter from the form
        selected_letter = request.form.get("alphabet")
        if selected_letter:
            jpg_path = f"/static/images/{language}/{selected_letter}.jpg"
            jpeg_path = f"/static/images/{language}/{selected_letter}.jpeg"
            image_path = None
            
            if os.path.exists(f".{jpg_path}"):  # Check for .jpg file
                image_path = jpg_path
            elif os.path.exists(f".{jpeg_path}"):  # Check for .jpeg file
                image_path = jpeg_path

            # Return JSON response with the image path and selected letter
            return jsonify({"image_path": image_path, "selected_letter": selected_letter})

        # Handle image file upload
        uploaded_file = request.files.get("image")  # Assuming the input name is "image"
        if uploaded_file:
            model = get_model_for_language(language)  # Function to retrieve the model for the selected language
            encoder = get_encoder_for_language(language)  # Function to retrieve the encoder for the selected language

            # Call the get_image_prediction function
            predicted_label= get_image_prediction(uploaded_file, model, encoder)

            # Return JSON response with prediction result
            return jsonify({"predicted_label": predicted_label})

    # Render the full template for GET requests
    return render_template(
        "classification.html",
        show_result=False,
        language=language,
        alphabet=alphabet,
        image_path=None
    )



@app.route("/name/<language>", methods=['GET', 'POST'])
def handle_name(language):
    if request.method == 'POST':
        if 'name' in request.form:  # Handle name input
            name = request.form.get("name", "").lower()  # Get name and convert to lowercase
            # Validate name input
            if not name.isalpha() or len(name) < 2:  # Check if name is valid
                return jsonify({"error": "Invalid name"}), 400

            characters = list(name)  # Get individual characters
            images = []

            # Collect image paths for each character
            for char in characters:
                char_image_path_jpg = f"/static/images/{language}/{char}.jpg"
                char_image_path_jpeg = f"/static/images/{language}/{char}.jpeg"

                # Check for the existence of image files
                if os.path.exists(f".{char_image_path_jpg}"):
                    images.append(char_image_path_jpg)
                elif os.path.exists(f".{char_image_path_jpeg}"):
                    images.append(char_image_path_jpeg)
                else:
                    return jsonify({"error": f"No image found for character '{char}'"}), 404

            if not images:
                return jsonify({"error": "No images found"}), 404

            return jsonify({"images": images, "letters": characters})

        # Handle image upload for prediction
        uploaded_file = request.files.get("image")
        if uploaded_file:
            # print("Uploaded file name:", uploaded_file.filename)  # Debugging line
            model = get_model_for_language(language)
            encoder = get_encoder_for_language(language)

            predicted_label = get_image_prediction(uploaded_file, model, encoder)

            return jsonify({"predicted_label": predicted_label})

        return jsonify({"error": "No valid data provided"}), 400

    return jsonify({"error": "Invalid request method"}), 405



@app.route('/cross_language')
def cross_language():
    return render_template('cross_language.html')

# Convert keys to lowercase
phonetic_mapping_lower = {
    lang.lower(): {letter.lower(): translations for letter, translations in letters.items()}
    for lang, letters in consts.phonetic_mapping.items()
}


@app.route('/get_mapping', methods=['GET'])
def get_mapping():
    source_language = request.args.get('sourceLanguage', '').lower()
    target_language = request.args.get('targetLanguage', '').lower()
    letter = request.args.get('letter', '').lower()

    # print(f"Received GET request with source_language={source_language}, target_language={target_language}, letter={letter}")

    # Check if the source language exists in the mapping
    if source_language in phonetic_mapping_lower:
        # Check if the letter exists for that language
        if letter in phonetic_mapping_lower[source_language]:
            letter_mapping = {lang.lower(): translation.lower() for lang, translation in phonetic_mapping_lower[source_language][letter].items()}
            # print(f"Letter mapping found: {letter_mapping}")

            # Check if target language is in the letter mapping
            if target_language in letter_mapping:
                # Get the mapped letter and ensure it's in lowercase
                mapped_letter = letter_mapping[target_language]
                # print(f"Mapped letter for target language '{target_language}': {mapped_letter}")
                return jsonify({"letter": letter, "target_language": target_language, "mapped_letter": mapped_letter})
            else:
                # print(f"No mapping found for target language '{target_language}' in letter mapping.")
                return jsonify({"message": f"No mapping found for target language '{target_language}'"})
        else:
            # print(f"Letter '{letter}' not found in source language '{source_language}' mapping.")
            return jsonify({"message": f"Letter '{letter}' not found in source language '{source_language}' mapping."})
    else:
        # print(f"Source language '{source_language}' not found in phonetic mapping.")
        return jsonify({"message": f"Source language '{source_language}' not found in phonetic mapping."})

@app.route('/get_image', methods=['POST'])
def get_image():
    data = request.get_json()
    # print(f"POST Data: {data}")  # Log the incoming POST data
    source_language = data.get('language', '').lower()
    letter = data.get('letter', '').lower()
    char_image_path_jpg = f"static/images/{source_language}/{letter}.jpg"
    char_image_path_jpeg = f"static/images/{source_language}/{letter}.jpeg"

    if os.path.exists(char_image_path_jpg):
        return jsonify({"image": f"/{char_image_path_jpg}"})
    elif os.path.exists(char_image_path_jpeg):
        return jsonify({"image": f"/{char_image_path_jpeg}"})
    else:
        return jsonify({"message": "No image found for the mapped letter"})


@app.route('/sign_to_sign')
def sign_to_sign():
    return render_template('sign_to_sign.html')

@app.route("/sign/<source_language>", methods=['GET', 'POST'])
def sign(source_language):
    target_language = request.args.get('targetLanguage', '').lower()
    if request.method == "POST":
        # Handle image file upload
        uploaded_file = request.files.get("image")  # Assuming the input name is "image"
        if uploaded_file:
            # Retrieve model and encoder for the language
            model = get_model_for_language(source_language)  # Function to retrieve the model for the selected language
            encoder = get_encoder_for_language(source_language)  # Function to retrieve the encoder for the selected language
            predicted_letter = get_image_prediction(uploaded_file, model, encoder)
            # Use the predicted letter to find its phonetic mapping
            letter = predicted_letter.lower()
            # Initialize mappings
            mapped_letters = []  # To store multiple mappings if they exist
            if source_language in phonetic_mapping_lower:
                if letter in phonetic_mapping_lower[source_language]:
                    letter_mapping = {lang.lower(): translation.lower() for lang, translation in phonetic_mapping_lower[source_language][letter].items()}
                    # Check if the target language exists in the letter mapping
                    if target_language in letter_mapping:
                        # Get the letter mapping for the target language
                        mapping_value = letter_mapping[target_language].strip()
                        # Split the mapping value by '/' and filter out any empty strings
                        if '/' in mapping_value:
                            mapped_letters = [item.strip() for item in mapping_value.split('/') if item.strip()]
                        else:
                            # If it’s a single letter, just wrap it in a list
                            mapped_letters = mapping_value


            # If no mappings are found, you might want to handle that
            if not mapped_letters:
                print("No mapping")  # Default to predicted letter if no mapping found

            # Fetch all possible image paths for the mapped letters
            image_paths = []
            for mapped_letter in mapped_letters:
                char_image_path_jpg = f"static/images/{target_language}/{mapped_letter}.jpg"
                char_image_path_jpeg = f"static/images/{target_language}/{mapped_letter}.jpeg"
                if os.path.exists(char_image_path_jpg):
                    image_paths.append(char_image_path_jpg)
                elif os.path.exists(char_image_path_jpeg):
                    image_paths.append(char_image_path_jpeg)


            # Return JSON response with the predicted label and all mapped image paths
            response_data = {
                "predicted_label": predicted_letter,
                "letter_mapping": mapped_letters,
                "image_paths": image_paths
            }

            return jsonify(response_data)

    # Render the full template for GET requests
    return render_template(
        "sign_to_sign.html",
        show_result=False,
        language=source_language,
        image_paths=None
    )




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)