import cv2
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort  # 트래킹을 위한 라이브러리
from age_model import ResNetAgeModel, device, test_transform
from gender_model import ResNetGenderModel, device, test_transform
from PIL import Image
from collections import Counter
from torchvision import transforms
from PIL import Image
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        hidden_channels = out_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.cv2 = Conv(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks):
        super(C2f, self).__init__()
        hidden_channels = out_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.cv2 = Conv(hidden_channels * (num_bottlenecks + 1), out_channels, kernel_size=1, stride=1, padding=0)
        self.bottlenecks = nn.ModuleList([Bottleneck(hidden_channels, hidden_channels) for _ in range(num_bottlenecks)])

    def forward(self, x):
        y = [self.cv1(x)]
        for bottleneck in self.bottlenecks:
            y.append(bottleneck(y[-1]))
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.cv2 = Conv(hidden_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class Detect(nn.Module):
    def __init__(self, num_classes=1):
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            Conv(1280, 640, 3, 1, 1),
            Conv(640, 320, 3, 1, 1),
            nn.Conv2d(320, num_classes + 5, 1, 1)
        )

    def forward(self, x):
        return self.conv(x)

class YOLOvBIT(nn.Module):
    def __init__(self, num_classes=1):
        super(YOLOvBIT, self).__init__()
        self.conv1 = Conv(3, 80, 3, 2, 1)  # P1/2
        self.conv2 = Conv(80, 160, 3, 2, 1)  # P2/4
        self.c2f1 = C2f(160, 160, 3)  # num_bottlenecks: 3
        self.conv3 = Conv(160, 320, 3, 2, 1)  # P3/8
        self.c2f2 = C2f(320, 320, 6)  # num_bottlenecks: 6
        self.conv4 = Conv(320, 640, 3, 2, 1)  # P4/16
        self.c2f3 = C2f(640, 640, 6)  # num_bottlenecks: 6
        self.conv5 = Conv(640, 1280, 3, 2, 1)  # P5/32
        self.c2f4 = C2f(1280, 1280, 3)  # num_bottlenecks: 3
        self.sppf = SPPF(1280, 1280)  # P5/32
        self.detect = Detect(num_classes)  # Detection layer

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.c2f1(x2)
        x4 = self.conv3(x3)
        x5 = self.c2f2(x4)
        x6 = self.conv4(x5)
        x7 = self.c2f3(x6)
        x8 = self.conv5(x7)
        x9 = self.c2f4(x8)
        x10 = self.sppf(x9)
        outputs = self.detect(x10)
        return outputs

# 커스텀 모델 사용 시 필요
def load_model(model_path, device):
    model = YOLOvBIT(num_classes=1)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# 커스텀 모델 사용 시 필요
def detect_objects(model, image, device, min_size=40, max_size=500):
    print("Detecting persons...")

    # 이미지가 numpy 배열인 경우 PIL 이미지로 변환
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_image)
    
    person_detections = []
    outputs = outputs.cpu().squeeze().permute(1, 2, 0)
    grid_size = outputs.shape[0]

    for y in range(grid_size):
        for x in range(grid_size):
            confidence = outputs[y, x, 4].item()

            # 신뢰도가 일정 수준 이상일 때만 바운딩 박스를 추출
            if confidence >= 0.5:
                bbox = outputs[y, x, :4].numpy()
                cx, cy, w, h = bbox
                cx *= image.width
                w *= image.width
                cy *= image.height
                h *= image.height
                # 그리드 좌표를 이미지 크기에 맞게 변환
                x_min = int(cx - w / 2)
                y_min = int(cy - h / 2)
                x_max = int(cx + w / 2)
                y_max = int(cy + h / 2)
                
                # 너무 작거나 너무 큰 객체는 필터링
                if w >= min_size and h >= min_size and w <= max_size and h <= max_size:
                    person_detections.append((x_min, y_min, x_max, y_max))

    return person_detections

def load_detections_and_tracks(image_name, bbox_txt_file):
    person_detections = []
    tracker_outputs = []

    with open(bbox_txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0] == image_name:
                track_id = int(parts[1])
                x1, y1, x2, y2 = map(int, parts[2:])
                person_detections.append([x1, y1, x2, y2])
                tracker_outputs.append(track_id)

    return person_detections, tracker_outputs

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
        yolo_results = yolo_model.predict(source=[frame], save=False, classes=[0])[0]
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

    def recognize_faces(self, image_file, frame, frame_number, image_name, gender_model, age_model, upclothes_model, downclothes_model, output_dir, bbox_txt_file):
        original_shape = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        forward_embeddings = self.extract_embeddings(frame_rgb)

        # 바운딩 박스와 track_id 정보를 텍스트 파일에서 불러옴
        person_detections, tracker_outputs = load_detections_and_tracks(image_file, bbox_txt_file)

        predictions = []

        for bbox, track_id in zip(person_detections, tracker_outputs):
            x_min, y_min, x_max, y_max = bbox
            bbox = [x_min, y_min, x_max, y_max]
            bbox = [int(coord) for coord in np.array(bbox)]  # 모든 좌표를 int형으로 변환
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

def process_images(image_dir, yolo_model_path, gender_model_path, age_model_path, upclothes_model_path, downclothes_model_path, output_dir, log_file, output_txt_path, bbox_txt_file):
    recognizer = FaceRecognizer()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gender_model = YOLO(gender_model_path, device)
    age_model = ResNetAgeModel(num_classes=4)
    age_model.load_state_dict(torch.load(age_model_path))
    age_model = age_model.to(device)
    age_model.eval()

    upclothes_model = load_yolo_model(upclothes_model_path, device)
    downclothes_model = load_yolo_model(downclothes_model_path, device)

    # 처리된 이미지 추적
    processed_images = load_processed_images(log_file)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    all_predictions = []

    for index, image_file in enumerate(image_files, start=1):
        if image_file in processed_images:
            #print(f"이미 처리된 이미지: {image_file}, 스킵합니다.")
            continue

        #if index % 6 != 0:
            #print(f"스킵되는 이미지: {image_file}, 처리하지 않습니다.")
            #continue

        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)

        predictions = recognizer.recognize_faces(image_file, frame, index, image_file, gender_model, age_model, upclothes_model, downclothes_model, output_dir, bbox_txt_file)
        save_predictions_to_txt(predictions, output_txt_path)

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
        #user_id, user_cam_folder_path, origin_filepath
        user_id = sys.argv[1] #horyunee / sys.argv[1]
        user_cam_folder_path = sys.argv[2] #./realtime_saved_images/horyunee/WB39 / sys.argv[2]
        image_directory = sys.argv[3] #./realtime_saved_images/horyunee/WB39/processed_images / sys.argv[3]
        yolo_model_path = './models/yolovBIT_170.pt' # 커스텀 모델 사용 시 필요
        gender_model_path = './models/gender_model.pt'
        age_model_path = './models/age_model.pth'
        #color_model_path = './models/color_model.pt'
        upclothes_model_path = './models/best_Version3_top.pt'
        downclothes_model_path = './models/best_Version3_bottom.pt'
        
        output_dir = os.path.join(user_cam_folder_path, "cropped_images").replace("\\", "/")
        
        output_txt_path = os.path.join(user_cam_folder_path, "predictions.txt").replace("\\", "/")
        log_file = os.path.join(user_cam_folder_path, "processed_images.log").replace("\\", "/")
        bbox_txt_file = os.path.join(image_directory, "bbox_info.txt").replace("\\", "/")
        
        predictions = process_images(image_directory, yolo_model_path, gender_model_path, age_model_path, upclothes_model_path, downclothes_model_path, output_dir, log_file, output_txt_path, bbox_txt_file)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
