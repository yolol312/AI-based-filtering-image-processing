import cv2
import os
import sys
import torch
import numpy as np
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from age_model import ResNetAgeModel, device, test_transform
from gender_model import ResNetGenderModel, device , test_transform
from PIL import Image
from collections import Counter

# 전역 변수로 선언하여 여러 비디오 처리 시 유지되도록 설정
global_persons_by_user = {}
global_next_person_id = {}

class FaceRecognizer:
    def __init__(self, user_no, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.user_no = user_no
        self.persons = global_persons_by_user.get(user_no, {})
        self.age_predictions = {str(pid): {'frames': [], 'age': None} for pid in self.persons.keys()}
        self.face_frame_count = {}  # 얼굴이 발견된 프레임 수를 기록하는 딕셔너리
        
        # 기존 person_id 중 가장 큰 값에 +1을 하여 next_person_id로 설정
        existing_ids = [int(pid) for pid in self.persons.keys()]
        self.next_person_id = max(existing_ids) + 1 if existing_ids else 1
        global_next_person_id[self.user_no] = self.next_person_id  # 전역적으로 업데이트

    def update_person_id(self):
        global global_next_person_id
        self.next_person_id += 1
        global_next_person_id[self.user_no] = self.next_person_id  # 전역적으로 업데이트

    def detect_persons(self, frame, yolo_model):
        print("Detecting persons...")
        yolo_results = yolo_model.predict(source=[frame], save=False, classes=[0])[0]
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

    def update_person_embedding(self, person_id, new_embedding):
        #기존 평균 임베딩에 새로운 임베딩을 추가하여 평균을 업데이트
        try:
            person_data = self.persons[person_id]
            count = len(person_data['embeddings'])
            updated_embedding = (np.array(person_data['average_embedding']) * count + np.array(new_embedding)) / (count + 1)
            person_data['average_embedding'] = updated_embedding.tolist()
            person_data['embeddings'].append(new_embedding)
        except KeyError:
            print(f"Error: Person ID {person_id} not found in persons dictionary.")


    def recognize_faces(self, frame, frame_number, output_dir, video_name, yolo_model, gender_model, age_model, upclothes_model, downclothes_model):
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
                    similarity = self.compare_similarity(embedding, data['average_embedding'])
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match_id = person_id

                if highest_similarity > 0.6:
                    person_id = str(best_match_id)  # 일관되게 문자열로 변환
                    matched = True
                    self.update_person_embedding(person_id, embedding)
                    self.face_frame_count[person_id] = self.face_frame_count.get(person_id, 0) + 1
                else:
                    person_id = str(self.next_person_id)  # 일관되게 문자열로 변환
                    self.update_person_id()
                    self.persons[person_id] = {
                        'embeddings': [embedding],
                        'average_embedding': embedding.tolist()
                    }
                    self.age_predictions[person_id] = {'frames': [], 'age': None}
                    self.face_frame_count[person_id] = 1

                # Error 메시지 방지를 위한 확인
                if person_id not in self.age_predictions:
                    self.age_predictions[person_id] = {'frames': [], 'age': None}

                print(f"Person ID: {person_id}, Frame Count: {self.face_frame_count[person_id]}")

                # Get gender prediction
                gender = predict_gender(face_image, gender_model)

                # Get age prediction
                age = predict_age(face_image, age_model)
                self.age_predictions[person_id]['frames'].append(age)
                most_common_age = Counter(self.age_predictions[person_id]['frames']).most_common(1)[0][0]
                self.age_predictions[person_id]['age'] = most_common_age

                # Find the person detection box corresponding to this face box
                for person_box in person_detections:
                    px1, py1, px2, py2 = person_box
                    fx1, fy1, fx2, fy2 = face_box
                    if fx1 >= px1 and fx2 <= px2 and fy1 >= py1 and fy2 <= py2:

                        # Get upclothes prediction
                        upclothes = predict_upclothes(frame, upclothes_model, person_box)

                        # Get downclothes prediction
                        downclothes = predict_downclothes(frame, downclothes_model, person_box)

                        # Save face images only if they appear in at least 2 frames
                        if self.face_frame_count[person_id] >= 6:
                            output_folder = os.path.join(output_dir, f'{video_name}_face')
                            os.makedirs(output_folder, exist_ok=True)
                            face_image_resized = cv2.resize(face_image, (160, 160))
                            output_path = os.path.join(output_folder, f'person_{person_id}_frame_{frame_number}_gender_{gender}_age_{self.age_predictions[person_id]["age"]}_upclothes_{upclothes}_downclothes_{downclothes}.jpg')
                            cv2.imwrite(output_path, cv2.cvtColor(face_image_resized, cv2.COLOR_RGB2BGR))
                            print(f"Saved image: {output_path}, person ID: {person_id}, detection probability: {prob}")

        return frame


#동양인 모델
def predict_gender(face_image, gender_model):
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    results = gender_model.predict(source=[face_rgb], save=False)
    genders = {0: "Male", 1: "Female"}
    gender_id = results[0].boxes.data[0][5].item()
    return genders.get(gender_id, "Unknown")


#서양인 모델
def predict_gender2(face_image, gender_model):
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

def predict_upclothes(frame, upclothes_model, bbox):
    # 옷 종류 예측을 위한 관심 영역(ROI) 추출
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:  # ROI가 비어있다면 'unknown' 반환
        return "unknown"
    
    # 옷 종류 예측 수행
    upclothes_results = upclothes_model.predict(source=[roi], save=False)[0]
    
    # 옷 종류 이름 정의
    upclothes_names = {1: 'longsleevetop', 2: 'shortsleevetop', 3: 'sleeveless'}
    
    if upclothes_results.boxes.data.shape[0] > 0:
        # 예측된 옷 종류 클래스 추출
        upclothes_class = int(upclothes_results.boxes.data[0][5])
        return upclothes_names.get(upclothes_class, "unknown")
    
    # 예측이 없을 경우, 'unknown' 반환
    return "unknown"

def predict_downclothes(frame, downclothes_model, bbox):
    # 옷 종류 예측을 위한 관심 영역(ROI) 추출
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:  # ROI가 비어있다면 'unknown' 반환
        return "unknown"
    
    # 옷 종류 예측 수행
    downclothes_results = downclothes_model.predict(source=[roi], save=False)[0]
    
    # 옷 종류 이름 정의
    downclothes_names = {0: 'shorts', 1: 'pants', 2: 'skirt'}
    
    if downclothes_results.boxes.data.shape[0] > 0:
        # 예측된 옷 종류 클래스 추출
        downclothes_class = int(downclothes_results.boxes.data[0][5])
        return downclothes_names.get(downclothes_class, "unknown")
    
    # 예측이 없을 경우, 'unknown' 반환
    return "unknown"

# YOLO 모델을 GPU에서 실행하도록 수정
def load_yolo_model(model_path, device):
    model = YOLO(model_path)
    model.to(device)  # 모델을 GPU 또는 CPU로 이동
    return model

# global_persons 정보를 .txt 파일로 저장
def save_global_persons(user_no, output_dir, face_frame_count, min_frames = 6):
    global global_persons_by_user
    output_path = os.path.join(output_dir, f"{user_no}_global_persons.txt")
    
    # 새로운 딕셔너리 생성, 평균 임베딩만 저장
    persons_to_save = {}
    for person_id, person_data in global_persons_by_user[user_no].items():
        # person_id가 face_frame_count에서 최소 프레임 수를 만족하는지 확인
        if face_frame_count.get(person_id, 0) >= min_frames:
            # 먼저 average_embedding이 리스트가 아니라면 NumPy 배열로 변환
            if isinstance(person_data['average_embedding'], list):
                average_embedding = np.array(person_data['average_embedding'])
            else:
                average_embedding = person_data.get('average_embedding')

            persons_to_save[person_id] = {
                'average_embedding': average_embedding.tolist()  # NumPy 배열로 변환한 후 tolist() 호출
            }
        else:
            print(f"Person {person_id}는 {min_frames} 프레임 이하로 탐지되어 저장되지 않습니다.")

    if persons_to_save:
        # 기존 파일이 있으면 병합하여 업데이트
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                existing_persons = json.load(f)
                for person_id, person_data in persons_to_save.items():
                    if person_id in existing_persons:
                        # 기존 데이터에 average_embedding 업데이트
                        existing_embedding = np.array(existing_persons[person_id]['average_embedding'])
                        new_embedding = np.array(person_data['average_embedding'])
                        updated_embedding = (existing_embedding + new_embedding) / 2
                        existing_persons[person_id]['average_embedding'] = updated_embedding.tolist()
                    else:
                        # 새로 발견된 person_id 추가
                        existing_persons[person_id] = person_data
                persons_to_save = existing_persons

        with open(output_path, 'w') as f:
            json.dump(persons_to_save, f)
        print(f"저장된 person 데이터: {len(persons_to_save)}명")
    else:
        print("저장할 person 데이터가 없습니다.")

# global_persons 정보를 .txt 파일에서 불러오기
def load_global_persons(user_no, output_dir):
    global global_persons_by_user
    input_path = os.path.join(output_dir, f"{user_no}_global_persons.txt")
    
    if os.path.exists(input_path):
        with open(input_path, 'r') as f:
            loaded_persons = json.load(f)
            
            # 평균 임베딩을 리스트에서 NumPy 배열로 변환
            for person_id, person_data in loaded_persons.items():
                if 'average_embedding' in person_data:
                    person_data['average_embedding'] = np.array(person_data['average_embedding'])
                if 'embeddings' not in person_data:
                    person_data['embeddings'] = []

            global_persons_by_user[user_no] = loaded_persons

            # age_predictions도 초기화
            face_recognizer = FaceRecognizer(user_no)
            for person_id in loaded_persons.keys():
                face_recognizer.age_predictions[person_id] = {'frames': [], 'age': None}
    else:
        global_persons_by_user[user_no] = {}  # 파일이 없으면 빈 딕셔너리로 초기화

def process_video(video_path, output_dir, yolo_model_path, gender_model_path, age_model_path, upclothes_model_path, downclothes_model_path, user_no, global_persons={}):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    recognizer = FaceRecognizer(user_no)
    recognizer.persons = global_persons

    v_cap = cv2.VideoCapture(video_path)
    frame_rate = int(v_cap.get(cv2.CAP_PROP_FPS))
    frame_interval = 6 # 8프레임마다 처리

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_device = torch.device('cpu')

    yolo_model = load_yolo_model(yolo_model_path, device)
    #동양인 모델
    gender_model = YOLO(gender_model_path, device)

    #외국인 전용 모델
    #gender_model = ResNetGenderModel(num_classes=2)
    #gender_model.load_state_dict(torch.load(gender_model_path))
    #gender_model = gender_model.to(device)
    #gender_model.eval()

    age_model = ResNetAgeModel(num_classes=4)
    age_model.load_state_dict(torch.load(age_model_path))
    age_model = age_model.to(device)
    age_model.eval()

    upclothes_model = load_yolo_model(upclothes_model_path, device) 
    downclothes_model = load_yolo_model(downclothes_model_path, device)
    frame_number = 0

    while True:
        success, frame = v_cap.read()
        if not success:
            break
        frame_number += 1

        # 1280x720보다 크면 리사이즈
        #if frame.shape[1] > 1280 or frame.shape[0] > 720:
            #frame = cv2.resize(frame, (1280, 720))

        if frame_number % frame_interval != 0:
            continue

        frame = recognizer.recognize_faces(frame, frame_number, output_dir, video_name, yolo_model, gender_model, age_model, upclothes_model, downclothes_model)

        #cv2.imshow('Processed Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    v_cap.release()
    cv2.destroyAllWindows()

    # 디버깅 출력 추가
    print(f"Final face_frame_count: {recognizer.face_frame_count}")

    return recognizer.persons, recognizer.face_frame_count  # face_frame_count도 반환

# process_videos 함수에서 save_global_persons 호출 시 recognizer.face_frame_count를 올바르게 전달합니다.
def process_videos(video_paths, output_dir, yolo_model_path, gender_model_path, age_model_path, upclothes_model_path, downclothes_model_path, user_no, target_video_name):
    global global_persons_by_user
    
    if user_no not in global_persons_by_user:
        global_persons_by_user[user_no] = {}  # 해당 user_no에 대한 global_persons를 초기화
    
    # global_persons를 파일에서 불러오기
    load_global_persons(user_no, output_dir)
    
    recognizer = None
    for video_path in video_paths:
        # video_name 필터링 추가
        video_base_name = os.path.splitext(os.path.basename(video_path))[0]
        if video_base_name != target_video_name:
            continue  # video_name과 일치하지 않으면 건너뛰기

        # 해당 user_no의 global_persons를 전달하여 비디오 처리
        recognizer = FaceRecognizer(user_no)
        recognizer.persons = global_persons_by_user[user_no]
        global_persons_by_user[user_no], final_face_frame_count = process_video(
            video_path, output_dir, yolo_model_path, gender_model_path, age_model_path, upclothes_model_path, downclothes_model_path, user_no, global_persons_by_user[user_no]
        )

    # 처리된 global_persons를 파일에 저장 (recognizer가 None이 아닌지 확인하고 face_frame_count를 함께 전달)
    if recognizer:
        save_global_persons(user_no, output_dir, final_face_frame_count)

    # 디버깅 정보 출력
    print(f"Final face_frame_count passed to save_global_persons: {final_face_frame_count}")


if __name__ == "__main__":
    try:
        video_name = "W1-3"
        user_no =  "modelTest"

        #video_name = sys.argv[1]
        #user_no = sys.argv[2]
        video_directory = f"./uploaded_videos/{user_no}/"
        video_paths = [os.path.join(video_directory, file) for file in os.listdir(video_directory) if file.endswith(('.mp4', '.avi', '.mov'))]
        output_directory = f"./extracted_images/{user_no}/"
        yolo_model_path = './models/yolov8x.pt'
        
        #gender_model_path = './models/gender_best.pth'  #서양인 모델
        gender_model_path = './models/gender_model.pt' #동양인 모델
        age_model_path = './models/age_model_0814.pth'
        upclothes_model_path = './models/upclothes_model.pt'
        downclothes_model_path = './models/downclothes_model2.pt'
        
        process_videos(video_paths, output_directory, yolo_model_path, gender_model_path, age_model_path, upclothes_model_path, downclothes_model_path, user_no, video_name)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

