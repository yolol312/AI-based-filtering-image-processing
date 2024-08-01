import cv2
import sys
import os
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from age_model import ResNetAgeModel, device, test_transform
from collections import Counter
from PIL import Image

class FaceRecognizer:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.persons = {}
        self.age_predictions = {}
        self.face_frame_count = {}
        self.next_person_id = 1

    def extract_embeddings(self, image):
        boxes, probs = self.mtcnn.detect(image)
        if boxes is not None:
            embeddings = []
            for box, prob in zip(boxes, probs):
                if prob < 0.99:
                    continue
                box = [int(coord) for coord in box]
                box = self._expand_box(box, image.shape[:2])
                face = image[box[1]:box[3], box[0]:box[2]]
                if face.size == 0 or face.shape[0] < 60 or face.shape[1] < 60:
                    continue
                face_tensor = torch.tensor(face).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device) / 255.0
                face_resized = torch.nn.functional.interpolate(face_tensor, size=(160, 160), mode='bilinear')
                face_image = face_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
                face_image = (face_image * 255).astype(np.uint8)
                embedding = self.resnet(face_resized).detach().cpu().numpy().flatten()
                embeddings.append((embedding, box, face_image, prob))
            return embeddings
        else:
            return []

    def _expand_box(self, box, image_shape, expand_factor=1.2):
        center_x, center_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        width, height = box[2] - box[0], box[3] - box[1]
        new_width, new_height = int(width * expand_factor), int(height * expand_factor)
        
        new_x1 = max(center_x - new_width // 2, 0)
        new_y1 = max(center_y - new_height // 2, 0)
        new_x2 = min(center_x + new_width // 2, image_shape[1])
        new_y2 = min(center_y + new_height // 2, image_shape[0])
        
        return [new_x1, new_y1, new_x2, new_y2]

    @staticmethod
    def compare_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def recognize_faces(self, frame, frame_number, output_dir, image_name, gender_model, age_model):
        original_shape = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        forward_embeddings = self.extract_embeddings(frame_rgb)

        if forward_embeddings:
            for embedding, box, face_image, prob in forward_embeddings:
                matched = False
                highest_similarity = 0
                best_match_id = None

                for person_id, data in self.persons.items():
                    for saved_embedding in data['embeddings']:
                        similarity = self.compare_similarity(embedding, saved_embedding)
                        if similarity > highest_similarity:
                            highest_similarity = similarity
                            best_match_id = person_id

                if highest_similarity > 0.75:
                    person_id = best_match_id
                    matched = True
                    if person_id in self.face_frame_count:
                        self.face_frame_count[person_id] += 1
                    else:
                        self.face_frame_count[person_id] = 1
                else:
                    person_id = self.next_person_id
                    self.next_person_id += 1
                    self.persons[person_id] = {'embeddings': []}
                    self.age_predictions[person_id] = {'frames': [], 'age': None}
                    self.face_frame_count[person_id] = 1

                self.persons[person_id]['embeddings'].append(embedding)

                # Get gender prediction
                gender = predict_gender(face_image, gender_model)

                # Get age prediction
                if person_id not in self.age_predictions:
                    self.age_predictions[person_id] = {'frames': [], 'age': None}

                age = predict_age(face_image, age_model)
                self.age_predictions[person_id]['frames'].append(age)

                # Determine the most common age from all frames
                most_common_age = Counter(self.age_predictions[person_id]['frames']).most_common(1)[0][0]
                self.age_predictions[person_id]['age'] = most_common_age

                # Save face images only if they appear in at least 5 frames
                if self.face_frame_count[person_id] >= 5:
                    output_folder = os.path.join(output_dir, f'{image_name}_face')
                    os.makedirs(output_folder, exist_ok=True)
                    face_image_resized = cv2.resize(face_image, (160, 160))
                    output_path = os.path.join(output_folder, f'person_{person_id}_frame_{frame_number}_gender_{gender}_age_{age}.jpg')
                    cv2.imwrite(output_path, cv2.cvtColor(face_image_resized, cv2.COLOR_RGB2BGR))
                    print(f"Saved image: {output_path}, person ID: {person_id}, detection probability: {prob}")

        return frame


def predict_gender(face_image, gender_model):
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    results = gender_model.predict(source=[face_rgb], save=False)
    genders = {0: "Male", 1: "Female"}
    gender_id = results[0].boxes.data[0][5].item()
    return genders.get(gender_id, "Unknown")

def predict_age(face_image, age_model):
    if isinstance(face_image, np.ndarray):
        face_image = Image.fromarray(face_image)

    face_tensor = test_transform(face_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = age_model(face_tensor)
        pred = logit.argmax(dim=1, keepdim=True).cpu().numpy()

    age_group = {0: "Child", 1: "Youth", 2: "Middle", 3: "Old"}
    if pred[0][0] < len(age_group):
        return age_group[pred[0][0]]
    else:
        return "Unknown"

def create_video(image_paths, output_video_directory, fps=24):
    os.makedirs(output_video_directory, exist_ok=True)
    video_name = "realtime"
    video_output_path = os.path.join(output_video_directory, "realtime.mp4")
    frame_size = (640, 480)
    video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image at {image_path} could not be loaded for video creation.")
            continue
        image_resized = cv2.resize(image, frame_size)
        video_writer.write(image_resized)

    video_writer.release()
    print(f"Video saved to {video_output_path}")

def process_images(image_paths, output_image_directory, yolo_model_path, gender_model_path, age_model_path, output_video_directory, fps=24):
    recognizer = FaceRecognizer()
    yolo_model = YOLO(yolo_model_path)
    gender_model = YOLO(gender_model_path)
    age_model = ResNetAgeModel(num_classes=4)
    age_model.load_state_dict(torch.load(age_model_path))
    age_model = age_model.to(device)
    age_model.eval()

    num_images = len(image_paths) - (len(image_paths) % 24)
    image_paths = image_paths[:num_images]

    create_video(image_paths, output_video_directory, fps)

    frame_number = 1
    for i in range(0, len(image_paths), 4):
        image_path = image_paths[i]
        start_time = os.path.splitext(os.path.basename(image_path))[0]
        image_name = "realtime"
        image = cv2.imread(image_path)
        recognizer.recognize_faces(image, frame_number, output_image_directory, image_name, gender_model, age_model)
        frame_number += 1

if __name__ == "__main__":
    try:
        user_no = "test"
        image_directory = f"./saved_images/{user_no}/"
        image_paths = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith(('.jpg', '.jpeg', '.png'))]
        output_image_directory = f"./realtime_extracted_images/{user_no}/"
        output_video_directory = f"./realtime_extracted_videos/{user_no}/"
        yolo_model_path = './models/yolov8x.pt'
        gender_model_path = './models/gender_model.pt'
        age_model_path = './models/age_best.pth'
        
        process_images(image_paths, output_image_directory, yolo_model_path, gender_model_path, age_model_path, output_video_directory)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
