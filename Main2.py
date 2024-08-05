import cv2
import os
import sys
import torch
import numpy as np
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from age_model import ResNetAgeModel, device, test_transform
from gender_model import ResNetGenderModel, device , test_transform
from PIL import Image
from collections import Counter

class FaceRecognizer:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.persons = {}
        self.age_predictions = {}
        self.face_frame_count = {}  # 얼굴이 발견된 프레임 수를 기록하는 딕셔너리
        self.next_person_id = 1  # 다음에 사용할 person_id를 관리

    def detect_persons(self, frame, yolo_model):
        print("Detecting persons...")
        yolo_results = yolo_model.predict(source=[frame], save=False)[0]
        person_detections = [
            (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
            for data in yolo_results.boxes.data.tolist()
            if float(data[4]) >= 0.85 and int(data[5]) == 0
        ]
        return person_detections
    
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
                if face.size == 0:
                    continue
                if face.shape[0] < 40 or face.shape[1] < 40:
                    continue
                face_tensor = torch.tensor(face).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device) / 255.0
                face_resized = torch.nn.functional.interpolate(face_tensor, size=(160, 160), mode='bilinear')
                face_image = face_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
                face_image = (face_image * 255).astype(np.uint8)  # Convert to uint8
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

    def recognize_faces(self, frame, frame_number, output_dir, video_name, yolo_model, gender_model, age_model, color_model, clothes_model):
        original_shape = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        forward_embeddings = self.extract_embeddings(frame_rgb)
        person_detections = self.detect_persons(frame, yolo_model)

        if forward_embeddings:
            for embedding, face_box, face_image, prob in forward_embeddings:
                matched = False
                highest_similarity = 0
                best_match_id = None

                for person_id, data in self.persons.items():
                    for saved_embedding in data['embeddings']:
                        similarity = self.compare_similarity(embedding, saved_embedding)
                        if similarity > highest_similarity:
                            highest_similarity = similarity
                            best_match_id = person_id

                if highest_similarity > 0.6:  # 유사도 임계값
                    person_id = best_match_id
                    matched = True
                    # Increase the frame count for this person_id
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

                # Find the person detection box corresponding to this face box
                for person_box in person_detections:
                    px1, py1, px2, py2 = person_box
                    fx1, fy1, fx2, fy2 = face_box
                    if fx1 >= px1 and fx2 <= px2 and fy1 >= py1 and fy2 <= py2:
                        # Get color prediction
                        color = predict_color(frame, color_model, person_box)

                        # Get clothes prediction
                        clothes = predict_clothes(frame, clothes_model, person_box)

                        # Save face images only if they appear in at least 5 frames
                        if self.face_frame_count[person_id] >= 5:
                            output_folder = os.path.join(output_dir, f'{video_name}_face')
                            os.makedirs(output_folder, exist_ok=True)
                            face_image_resized = cv2.resize(face_image, (160, 160))
                            output_path = os.path.join(output_folder, f'person_{person_id}_frame_{frame_number}_gender_{gender}_age_{age}_color_{color}_clothes_{clothes}.jpg')
                            cv2.imwrite(output_path, cv2.cvtColor(face_image_resized, cv2.COLOR_RGB2BGR))
                            print(f"Saved image: {output_path}, person ID: {person_id}, detection probability: {prob}")

        return frame


def predict_gender(face_image, gender_model):
    if isinstance(face_image, np.ndarray):
        face_image = Image.fromarray(face_image)

    face_tensor = test_transform(face_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = gender_model(face_tensor)
        pred = logit.argmax(dim=1, keepdim=True).cpu().numpy()

    gender_dict = {0: "Male", 1: "Female"}
    if pred[0][0] < len(gender_dict):
        return gender_dict[pred[0][0]]
    else:
        return "Unknown"
    
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
    
def predict_color(frame, color_model, bbox):
    # 색상 예측을 위한 관심 영역(ROI) 추출
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:  # ROI가 비어있다면 'unknown' 반환
        return "unknown"
    
    # 색상 예측 수행
    color_results = color_model.predict(source=[roi], save=False)[0]
    
    # 색상 이름 정의
    color_names = {0: 'black', 1: 'white', 2: 'red', 3: 'yellow', 4: 'green', 5: 'blue', 6: 'brown'}
    
    if color_results.boxes.data.shape[0] > 0:
        # 예측된 색상 클래스 추출
        color_class = int(color_results.boxes.data[0][5])
        return color_names.get(color_class, "unknown")
    
    # 예측이 없을 경우, 'unknown' 반환
    return "unknown"

def predict_clothes(frame, clothes_model, bbox):
    # 옷 종류 예측을 위한 관심 영역(ROI) 추출
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:  # ROI가 비어있다면 'unknown' 반환
        return "unknown"
    
    # 옷 종류 예측 수행
    clothes_results = clothes_model.predict(source=[roi], save=False)[0]
    
    # 옷 종류 이름 정의
    clothes_names = {0: 'dress', 1: 'longsleevetop', 2: 'shortsleevetop', 3: 'vest', 4: 'shorts', 5: 'pants', 6: 'skirt'}
    
    if clothes_results.boxes.data.shape[0] > 0:
        # 예측된 옷 종류 클래스 추출
        clothes_class = int(clothes_results.boxes.data[0][5])
        return clothes_names.get(clothes_class, "unknown")
    
    # 예측이 없을 경우, 'unknown' 반환
    return "unknown"

def process_video(video_path, output_dir, yolo_model_path, gender_model_path, age_model_path, color_model_path, clothes_model_path, global_persons={}):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    recognizer = FaceRecognizer()
    recognizer.persons = global_persons

    v_cap = cv2.VideoCapture(video_path)
    frame_rate = int(v_cap.get(cv2.CAP_PROP_FPS))
    frame_interval = 2 # 8프레임마다 처리

    yolo_model = YOLO(yolo_model_path)
    # gender_model = YOLO(gender_model_path)

    gender_model = ResNetGenderModel(num_classes=2)
    gender_model.load_state_dict(torch.load(gender_model_path))
    gender_model = gender_model.to(device)
    gender_model.eval()

    age_model = ResNetAgeModel(num_classes=4)
    age_model.load_state_dict(torch.load(age_model_path))
    age_model = age_model.to(device)
    age_model.eval()

    color_model = YOLO(color_model_path)
    clothes_model = YOLO(clothes_model_path) 

    frame_number = 0

    while True:
        success, frame = v_cap.read()
        if not success:
            break
        frame_number += 1

        if frame_number % frame_interval != 0:
            continue

        frame = recognizer.recognize_faces(frame, frame_number, output_dir, video_name, yolo_model, gender_model, age_model, color_model, clothes_model)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    v_cap.release()
    cv2.destroyAllWindows()

    return recognizer.persons

def process_videos(video_paths, output_dir, yolo_model_path, gender_model_path, age_model_path, color_model_path, clothes_model_path):
    global_persons = {}
    for video_path in video_paths:
        global_persons = process_video(video_path, output_dir, yolo_model_path, gender_model_path, age_model_path, color_model_path, clothes_model_path, global_persons)

if __name__ == "__main__":
    try:
        user_no = "test5"
        video_directory = f"./uploaded_videos/{user_no}/"
        video_paths = [os.path.join(video_directory, file) for file in os.listdir(video_directory) if file.endswith(('.mp4', '.avi', '.mov'))]
        output_directory = f"./extracted_images/{user_no}/"
        yolo_model_path = './models/yolov8x.pt'
        gender_model_path = './models/gender_best.pth'
        age_model_path = './models/age_best.pth'
        color_model_path = './models/color_model.pt'
        clothes_model_path = './models/clothes_class.pt'
        
        process_videos(video_paths, output_directory, yolo_model_path, gender_model_path, age_model_path, color_model_path, clothes_model_path)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)