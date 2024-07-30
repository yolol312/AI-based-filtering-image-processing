import cv2
import sys
import os
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from age_model import ResNetAgeModel, device, test_transform

class FaceRecognizer:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.tracked_faces = {}  # 트래킹 ID와 face_id를 매핑할 딕셔너리
        self.known_faces = set()  # 이미 트래킹 중인 face_id를 저장하는 집합
        
    # 이미지에서 얼굴 임베딩을 추출하는 함수
    def extract_embeddings(self, image):
        boxes, probs = self.mtcnn.detect(image)
        if boxes is None or probs is None:
            return []  # 얼굴이 감지되지 않으면 빈 리스트 반환

        embeddings = []
        for box, prob in zip(boxes, probs):
            if prob < 0.99:
                continue
            box = [int(coord) for coord in box]
            face = image[box[1]:box[3], box[0]:box[2]]
            if face.size == 0:
                continue
            face_tensor = torch.tensor(face).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device) / 255.0
            face_resized = torch.nn.functional.interpolate(face_tensor, size=(160, 160), mode='bilinear')
            embedding = self.resnet(face_resized).detach().cpu().numpy().flatten()
            embeddings.append((embedding, box))
        return embeddings

    # 알려진 얼굴 이미지를 로드하는 함수
    def load_known_faces(self, image_paths):
        known_faces = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            embeddings = self.extract_embeddings(image)
            if embeddings:
                embedding, box = embeddings[0]
                face_id = os.path.splitext(os.path.basename(image_path))[0]  # 파일 이름을 ID로 사용
                known_faces.append({'embedding': embedding, 'box': box, 'id': face_id})
            else:
                print(f"No faces detected in image: {image_path}")
        return known_faces

    # 두 임베딩 간의 유사도를 비교하는 함수
    @staticmethod
    def compare_similarity(embedding1, embedding2):
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        print(f"Similarity 유사도 입니다.: {similarity}")
        return similarity

    # 두 박스의 IoU를 계산하는 함수
    def calculate_iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        xi1 = max(x1, x3)  # 교차하는 부분의 좌측 상단 x 좌표
        yi1 = max(y1, y3)  # 교차하는 부분의 좌측 상단 y 좌표
        xi2 = min(x2, x4)  # 교차하는 부분의 우측 하단 x 좌표
        yi2 = min(y2, y4)  # 교차하는 부분의 좌측 상단 y 좌표

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)  # 교차하는 부분의 넓이 계산
        bbox1_area = (x2 - x1) * (y2 - y1)  # bbox1의 넓이 계산
        bbox2_area = (x4 - x3) * (y4 - y3)  # bbox2의 넓이 계산

        iou = inter_area / float(bbox1_area + bbox2_area - inter_area)  # IoU 계산

        overlap_coords = (xi1, yi1, xi2, yi2)  # 교차하는 부분의 좌표

        return iou, overlap_coords

    # 얼굴 임베딩을 비교하여 face_id를 할당하는 함수
    def assign_face_id(self, face_encoding, known_faces, threshold=0.68):
        if known_faces:
            distances = [self.compare_similarity(face_encoding, face['embedding']) for face in known_faces]
            min_distance = np.min(distances)
            min_distance_index = np.argmin(distances)

            if min_distance > threshold:
                return known_faces[min_distance_index]['id']
        return None

    # 프레임에서 사람을 감지하는 함수
    def detect_persons(self, frame, yolo_model):
        yolo_results = yolo_model.predict(source=[frame], save=False)[0]
        person_detections = [
            (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
            for data in yolo_results.boxes.data.tolist()
            if float(data[4]) >= 0.85 and int(data[5]) == 0
        ]
        return person_detections

    # 프레임에 바운딩 박스를 그리는 함수
    def draw_bounding_boxes(self, frame, tracks, tracked_faces):
        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()
            track_bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))

            if track.track_id in tracked_faces:
                face_id = tracked_faces[track.track_id]
                id_text = f"face_id: {face_id}"
                cv2.rectangle(frame, (track_bbox[0], track_bbox[1]), (track_bbox[2], track_bbox[3]), (0, 0, 255), 2)  # 특정 객체 빨간색 

    # 얼굴 인식 및 트래킹을 처리하는 함수
    def recognize_faces(self, frame, frame_number, output_dir, known_faces, tracker, video_name, yolo_model, gender_model, age_model):
        original_shape = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 얼굴 인식을 건너뛸지 여부를 결정하는 플래그
        skip_face_detection = False
        
        # 기존 트랙에서 이미 트래킹된 얼굴이 있는지 확인
        for track_id, face_id in self.tracked_faces.items():
            if track_id in [track.track_id for track in tracker.update_tracks([], frame=frame)]:
                skip_face_detection = True
                break

        # 얼굴 인식을 건너뛰지 않는 경우에만 얼굴을 인식
        if not skip_face_detection:
            detect_faces = self.extract_embeddings(frame_rgb)
        else:
            detect_faces = []

        person_detections = self.detect_persons(frame, yolo_model)
        results = []
        for (xmin, ymin, xmax, ymax) in person_detections:
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], 1.0, 0])

        tracks = tracker.update_tracks(results, frame=frame)

        active_track_ids = {track.track_id for track in tracks if track.is_confirmed()}
        inactive_track_ids = set(self.tracked_faces.keys()) - active_track_ids
        for inactive_track_id in inactive_track_ids:
            if self.tracked_faces[inactive_track_id] in self.known_faces:
                self.known_faces.remove(self.tracked_faces[inactive_track_id])
            del self.tracked_faces[inactive_track_id]

        overlapping_track_ids = set()
        frame_area = frame.shape[0] * frame.shape[1]

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            track_bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))

            bbox_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])

            if bbox_area / frame_area <= 0.021:
                if track_id in self.tracked_faces:
                    if self.tracked_faces[track_id] in self.known_faces:
                        self.known_faces.remove(self.tracked_faces[track_id])
                    del self.tracked_faces[track_id]
                continue
            
            # 얼굴 중심과 트랙 박스가 겹치는지 확인
            for embedding, box in detect_faces:
                left, top, right, bottom = box
                face_center_x = (left + right) / 2
                face_center_y = (top + bottom) / 2
                
                center_box_left = face_center_x - (right - left) / 6
                center_box_top = face_center_y - (bottom - top) / 6
                center_box_right = face_center_x + (right - left) / 6
                center_box_bottom = face_center_y + (bottom - top) / 6

                if (track_bbox[0] <= center_box_left <= track_bbox[2] or track_bbox[0] <= center_box_right <= track_bbox[2]) and \
                (track_bbox[1] <= center_box_top <= track_bbox[3] or track_bbox[1] <= center_box_bottom <= track_bbox[3]):
                    face_id = self.assign_face_id(embedding, known_faces)  # 얼굴 임베딩을 사용해 face_id 할당
                    if face_id is not None:
                        if face_id in self.known_faces:
                            # 이미 인식된 얼굴의 경우 빨간색 박스로 표시
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                            continue
                        # 새로운 얼굴을 인식한 경우 tracked_faces에 track_id와 face_id 매핑 추가
                        self.tracked_faces[track_id] = face_id
                        # known_faces에 face_id 추가
                        self.known_faces.add(face_id)
                    break  # 매핑이 완료되면 루프 종료
                
        self.draw_bounding_boxes(frame, tracks, self.tracked_faces)

        return frame


# 비디오를 처리하는 함수
def process_video(video_path, output_dir, known_face_paths, yolo_model_path, gender_model_path, age_model_path, person_id, target_fps=10, max_age=30, n_init=3, nn_budget=60):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    recognizer = FaceRecognizer()

    v_cap = cv2.VideoCapture(video_path)
    if not v_cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_rate = int(v_cap.get(cv2.CAP_PROP_FPS))
    frame_width, frame_height = None, None
    video_writer = {}

    known_faces = recognizer.load_known_faces(known_face_paths)
    yolo_model = YOLO(yolo_model_path)
    gender_model = YOLO(gender_model_path)
    age_model = ResNetAgeModel(num_classes=4)
    age_model.load_state_dict(torch.load(age_model_path))
    age_model = age_model.to(device)
    age_model.eval()

    tracker = DeepSort(max_age=max_age, n_init=n_init, nn_budget=nn_budget)

    frame_number = 0

    while True:
        success, frame = v_cap.read()
        if not success:
            break
        frame_number += 1

        frame = recognizer.recognize_faces(frame, frame_number, output_dir, known_faces, tracker, video_name, yolo_model, gender_model, age_model)

        if frame_width is None or frame_height is None:
            frame_height, frame_width = frame.shape[:2]
            output_person_folder = os.path.join(output_dir, f"{video_name}_clip", person_id)
            os.makedirs(output_person_folder, exist_ok=True)
            output_video_path = os.path.join(output_person_folder, f"{person_id}_output.mp4")
            if person_id not in video_writer:
                video_writer[person_id] = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (frame_width, frame_height))

        if person_id in video_writer:
            video_writer[person_id].write(frame)

        cv2.imshow('Processed Frame', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    v_cap.release()
    for writer in video_writer.values():
        writer.release()
    cv2.destroyAllWindows()


# 여러 비디오를 처리하는 함수
def process_videos(video_directory, output_dir, known_face_paths, yolo_model_path, gender_model_path, age_model_path, target_fps=10):
    video_paths = [os.path.join(video_directory, file) for file in os.listdir(video_directory) if file.endswith(('.mp4', '.avi', '.mov'))]
    
    target_recognizer = FaceRecognizer()
    target_faces = target_recognizer.load_known_faces(known_face_paths)
    if not target_faces:
        print("No known faces found.")
        return
    target_embedding = target_faces[0]['embedding']

    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        clip_folder = os.path.join(output_dir, f"{video_name}_clip")
        if not os.path.exists(clip_folder):
            continue
        person_folders = [d for d in os.listdir(clip_folder) if os.path.isdir(os.path.join(clip_folder, d))]
        for person_id in person_folders:
            person_folder_path = os.path.join(clip_folder, person_id)
            known_face_paths = [os.path.join(person_folder_path, img_file) for img_file in os.listdir(person_folder_path) if img_file.lower().endswith(('.png', '.jpg', '.jpeg'))]
            known_faces = target_recognizer.load_known_faces(known_face_paths)
            if not known_faces:
                print(f"No embeddings found for faces in {person_folder_path}.")
                continue
            for face in known_faces:
                similarity = target_recognizer.compare_similarity(target_embedding, face['embedding'])
                print(f"Comparing with {face['id']} - Similarity: {similarity}")
                if similarity >= 0.64:
                    process_video(video_path, output_dir, known_face_paths, yolo_model_path, gender_model_path, age_model_path, person_id, target_fps)
                    break

if __name__ == "__main__":
    try:
        # 스크립트 실행에 필요한 경로 및 파일 설정
        user_id = 'hyojin'  # 사용자의 ID를 설정하세요.
        video_directory = f"./uploaded_videos/{user_id}/"
        image_directory = f"./uploaded_images/{user_id}/"
        output_directory = f"./extracted_images/{user_id}/"
        yolo_model_path = './models/yolov8x.pt'
        gender_model_path = './models/gender_model.pt'
        age_model_path = './models/age_best.pth'

        # 이미지 디렉토리의 모든 이미지 파일 경로 읽기
        known_face_paths = [os.path.join(image_directory, img_file) for img_file in os.listdir(image_directory) if img_file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        process_videos(video_directory, output_directory, known_face_paths, yolo_model_path, gender_model_path, age_model_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
