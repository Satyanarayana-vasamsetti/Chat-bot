from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import pyttsx3
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
import speech_recognition as sr
import os
import cv2

# Set PyTorch settings to avoid cuDNN errors
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Initialize the Flask app
app = Flask(__name__)

# Ensure the uploads and outputs folders exist
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize the BLIP model for VQA
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cpu")

# Function to detect objects in an image
def detect_objects_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Perform inference
    results = model(img)

    detected_objects = []

    # Draw bounding boxes and labels on the image
    for result in results:
        for bbox in result.boxes:
            xyxy = bbox.xyxy[0]
            x1, y1, x2, y2 = map(int, xyxy)
            confidence = bbox.conf[0].item()
            class_id = bbox.cls[0].item()
            class_name = model.names[int(class_id)]
            detected_objects.append(f'{class_name} with confidence {confidence:.2f}')
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the annotated image
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    return output_path, detected_objects

# Function to detect objects in a video
def detect_objects_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    detected_objects = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)

        # Draw bounding boxes and labels on the frame
        for result in results:
            for bbox in result.boxes:
                xyxy = bbox.xyxy[0]
                x1, y1, x2, y2 = map(int, xyxy)
                confidence = bbox.conf[0].item()
                class_id = bbox.cls[0].item()
                class_name = model.names[int(class_id)]
                detected_objects.append(f'{class_name} with confidence {confidence:.2f}')
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{class_name}: {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    return output_path,detected_objects

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_path, detected_objects = detect_objects_image(file_path)
        elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            output_path, detected_objects = detect_objects_video(file_path)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400

        return jsonify({'output_file': os.path.basename(output_path), 'detected_objects': detected_objects})

# Route to serve output files
@app.route('/outputs/<filename>')
def send_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# Route to capture image from webcam and detect objects
@app.route('/webcam', methods=['POST'])
def webcam_capture():
    img_data = request.files['webcam_image']
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_capture.jpg')
    img_data.save(img_path)

    output_path, detected_objects = detect_objects_image(img_path)

    return jsonify({'output_file': os.path.basename(output_path), 'detected_objects': detected_objects})

# Route to ask a question via voice input
@app.route('/ask', methods=['POST'])
def ask_question():
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()

    try:
        with sr.Microphone() as source:
            print("Please ask your question...")
            engine.say("Please ask your question.")
            engine.runAndWait()
            audio = recognizer.listen(source)

            try:
                question = recognizer.recognize_google(audio)
                print(f"Question recognized: {question}")

                output_file = request.json.get('output_file')
                if not output_file:
                    return jsonify({'error': 'No output file provided'}), 400

                image_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)
                if not os.path.exists(image_path):
                    return jsonify({'error': 'Output file not found'}), 400

                # Load the image
                raw_image = Image.open(image_path).convert('RGB')

                # Prepare inputs for BLIP model
                inputs = processor(raw_image, question, return_tensors="pt").to("cpu")

                # Generate the answer
                out = vqa_model.generate(**inputs, max_new_tokens=50)
                answer = processor.decode(out[0], skip_special_tokens=True)

                print(f"Answer: {answer}")
                # engine.say(answer)
                engine.runAndWait()

                return jsonify({'question': question, 'answer': answer})

            except sr.UnknownValueError:
                return jsonify({'error': 'Could not understand the audio'}), 400

            except sr.RequestError as e:
                return jsonify({'error': f'Could not request results; {e}'}), 500

            except Exception as e:
                print(f"Error processing question: {e}")
                return jsonify({'error': str(e)}), 500

    except Exception as e:
        print(f"Error initializing microphone or recognizer: {e}")
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)