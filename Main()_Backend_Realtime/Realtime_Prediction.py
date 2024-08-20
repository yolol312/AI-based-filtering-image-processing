import cv2
import os
import sys
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort  # 트래킹을 위한 라이브러리
from age_model import ResNetAgeModel, device, test_transform
from gender_model import ResNetGenderModel, device, test_transform
from PIL import Image
from collections import Counter

class FaceRecognizer:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.persons = {}
        self.age_predictions = {}
        self.face_frame_count = {}
        self.next_person_id = 1

    def detect_persons(self, frame, yolo_model):
        print("Detecting persons...")
        yolo_results = yolo_model.predict(source=[frame], save=False)[0]
        person_detections = [
            (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
            for data in yolo_results.boxes.data.tolist()
            if float(data[4]) >= 0.85 and int(data[5]) == 0  # 사람(class=0)만 필터링
        ]
        return person_detections

    def extract_embeddings(self, image):
        print("Extract_embeddings...")
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
                if face.shape[0] < 60 or face.shape[1] < 60:
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

    def recognize_faces(self, frame, frame_number, image_name, yolo_model, gender_model, age_model, upclothes_model, downclothes_model, tracker, output_dir):
        original_shape = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        forward_embeddings = self.extract_embeddings(frame_rgb)
        person_detections = self.detect_persons(frame, yolo_model)

        results = []
        for (xmin, ymin, xmax, ymax) in person_detections:
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], 1.0, 0])

        tracker_outputs = tracker.update_tracks(results, frame=frame)

        predictions = []

        for track in tracker_outputs:
            bbox = track.to_tlbr()
            bbox = [int(coord) for coord in np.array(bbox)]  # 모든 좌표를 int형으로 변환
            track_id = track.track_id
            upclothes = predict_upclothes(frame, upclothes_model, bbox)
            downclothes = predict_downclothes(frame, downclothes_model, bbox)

            # 바운딩 박스 크기만큼 이미지를 잘라서 저장
            person_crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            crop_filename = os.path.join(output_dir, f"Person_{track_id}.jpg")
            cv2.imwrite(crop_filename, person_crop)
            print(f"Saved cropped image: {crop_filename}")
            
            face_detected = False
            for embedding, face_box, face_image, prob in forward_embeddings:
                # 트래킹 박스와 얼굴 박스가 일치하는지 확인
                if bbox[0] <= face_box[0] <= bbox[2] and bbox[1] <= face_box[1] <= bbox[3]:
                    face_detected = True
                    matched = False
                    highest_similarity = 0
                    best_match_id = None

                    for person_id, data in self.persons.items():
                        for saved_embedding in data['embeddings']:
                            similarity = self.compare_similarity(embedding, saved_embedding)
                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                best_match_id = person_id

                    if highest_similarity > 0.6:
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

                    gender = predict_gender(face_image, gender_model)
                    age = predict_age(face_image, age_model)

                    if person_id not in self.age_predictions:
                        self.age_predictions[person_id] = {'frames': [], 'age': None}

                    self.age_predictions[person_id]['frames'].append(age)
                    most_common_age = Counter(self.age_predictions[person_id]['frames']).most_common(1)[0][0]
                    self.age_predictions[person_id]['age'] = most_common_age

                    predictions.append({
                        'track_id': track_id,
                        'person_id': person_id,
                        'image_name': image_name,
                        'gender': gender,
                        'age': age,
                        'upclothes': upclothes,
                        'downclothes': downclothes,
                    })
                    break

            if not face_detected:  # 얼굴이 감지되지 않은 경우
                predictions.append({
                    'track_id': track_id,
                    'person_id': None,  # 얼굴 인식되지 않음
                    'image_name': image_name,
                    'gender': "Unknown",
                    'age': "Unknown",
                    'upclothes': upclothes,
                    'downclothes': downclothes,
                })

        return predictions

def load_processed_images(log_file):
    """이미 처리된 이미지 목록을 로드"""
    if not os.path.exists(log_file):
        return set()
    
    with open(log_file, 'r') as f:
        processed_images = set(line.strip() for line in f)
    return processed_images

def save_processed_image(log_file, image_file):
    """처리된 이미지 파일명을 로그 파일에 저장"""
    with open(log_file, 'a') as f:
        f.write(f"{image_file}\n")

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

def predict_upclothes(frame, clothes_model, bbox):
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"
    
    clothes_results = clothes_model.predict(source=[roi], save=False)[0]
    
    clothes_names = {0: 'longsleevetop', 1: 'shortsleevetop', 2: 'sleeveless'}
    
    if clothes_results.boxes.data.shape[0] > 0:
        clothes_class = int(clothes_results.boxes.data[0][5])
        return clothes_names.get(clothes_class, "unknown")
    
    return "unknown"

def predict_downclothes(frame, clothes_model, bbox):
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"
    
    clothes_results = clothes_model.predict(source=[roi], save=False)[0]
    
    clothes_names = {0: 'shorts', 1: 'pants', 2: 'skirt'}
    
    if clothes_results.boxes.data.shape[0] > 0:
        clothes_class = int(clothes_results.boxes.data[0][5])
        return clothes_names.get(clothes_class, "unknown")
    
    return "unknown"

def load_yolo_model(model_path, device):
    model = YOLO(model_path)
    model.to(device)
    return model

def process_images(image_dir, yolo_model_path, gender_model_path, age_model_path, upclothes_model_path, downclothes_model_path, output_dir, log_file, output_txt_path):
    recognizer = FaceRecognizer()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    yolo_model = load_yolo_model(yolo_model_path, device)
    gender_model = YOLO(gender_model_path, device)
    age_model = ResNetAgeModel(num_classes=4)
    age_model.load_state_dict(torch.load(age_model_path))
    age_model = age_model.to(device)
    age_model.eval()

    upclothes_model = load_yolo_model(upclothes_model_path, device)
    downclothes_model = load_yolo_model(downclothes_model_path, device)

    tracker = DeepSort(max_age=30, nn_budget=20)  # DeepSort 트래커 설정

    # 처리된 이미지 추적
    processed_images = load_processed_images(log_file)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    all_predictions = []

    for index, image_file in enumerate(image_files, start=1):
        if image_file in processed_images:
            #print(f"이미 처리된 이미지: {image_file}, 스킵합니다.")
            continue

        if index % 5 != 0:
            #print(f"스킵되는 이미지: {image_file}, 처리하지 않습니다.")
            continue

        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)

        predictions = recognizer.recognize_faces(frame, index, image_file, yolo_model, gender_model, age_model, upclothes_model, downclothes_model, tracker, output_dir)
        save_predictions_to_txt(predictions, output_txt_path)
        #all_predictions.extend(predictions)

        # 처리된 이미지 로그 저장
        save_processed_image(log_file, image_file)

    print("Image processing complete.")
    return all_predictions

def save_predictions_to_txt(predictions, output_file):
    with open(output_file, 'a') as f:
        for pred in predictions:
            f.write(f"Track ID: {pred['track_id']}, Person ID: {pred['person_id']}, Image: {pred['image_name']}, "
                    f"Gender: {pred['gender']}, Age: {pred['age']}, "
                    f"UpClothes: {pred['upclothes']}, DownClothes: {pred['downclothes']}\n")
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    try:
        #user_id, user_folder_path, origin_filepath
        user_id = sys.argv[1]
        user_folder_path = sys.argv[2]
        image_directory = sys.argv[3]
        yolo_model_path = './models/yolov8x.pt'
        gender_model_path = './models/gender_model.pt'
        age_model_path = './models/age_model.pth'
        #color_model_path = './models/color_model.pt'
        upclothes_model_path = './models/best_Version3_top.pt'
        downclothes_model_path = './models/best_Version3_bottom.pt'
        
        output_dir = os.path.join(user_folder_path, "cropped_images").replace("\\", "/")
        
        output_txt_path = os.path.join(user_folder_path, "predictions.txt").replace("\\", "/")
        log_file = os.path.join(user_folder_path, "processed_images.log").replace("\\", "/")
        
        predictions = process_images(image_directory, yolo_model_path, gender_model_path, age_model_path, upclothes_model_path, downclothes_model_path, output_dir, log_file, output_txt_path)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
