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
import subprocess

class FaceRecognizer:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.tracked_faces = {}
        self.previous_tracks = {}
        self.known_faces = {}
        self.inactive_faces = {}
        self.assigned_ids = set()

    def extract_embeddings(self, image):
        print("Extracting embeddings...")
        boxes, probs = self.mtcnn.detect(image)
        if boxes is None or probs is None:
            print("No faces detected.")
            return []

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
        print(f"Embeddings extracted: {len(embeddings)}")
        return embeddings

    def load_known_faces(self, image_paths):
        print("Loading known faces...")
        known_faces = []
        for image_path in image_paths:
            print(f"Processing image: {image_path}")
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            embeddings = self.extract_embeddings(image)
            if embeddings:
                embedding, box = embeddings[0]
                face_id = os.path.splitext(os.path.basename(image_path))[0]
                known_faces.append({'embedding': embedding, 'box': box, 'id': face_id})
                print(f"Face ID: {face_id} added.")
            else:
                print(f"No faces detected in image: {image_path}")
        print(f"Total known faces loaded: {len(known_faces)}")
        return known_faces

    @staticmethod
    def compare_similarity(embedding1, embedding2):
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        print(f"Similarity: {similarity}")
        return similarity

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

    def assign_face_id(self, face_encoding, known_faces, threshold=0.68):
        print("Assigning face ID...")
        if known_faces:
            distances = [self.compare_similarity(face_encoding, face['embedding']) for face in known_faces]
            max_similarity = np.max(distances)
            best_match_index = np.argmax(distances)

            if max_similarity > threshold:
                potential_face_id = known_faces[best_match_index]['id']
                if potential_face_id not in self.assigned_ids:
                    print(f"Assigned face ID: {potential_face_id}")
                    return potential_face_id
                else:
                    print(f"Face ID {potential_face_id} is already assigned.")
            else:
                print("No matching face ID found.")
        return None

    def detect_persons(self, frame, yolo_model, min_width=60, min_height=60):
        print("Detecting persons...")
        yolo_results = yolo_model.predict(source=[frame], save=False, classes=[0])[0]
        person_detections = []
        for data in yolo_results.boxes.data.tolist():
            xmin, ymin, xmax, ymax, conf, cls = int(data[0]), int(data[1]), int(data[2]), int(data[3]), float(data[4]), int(data[5])
        
            if conf >= 0.85 and cls == 0: 
                width = xmax - xmin
                height = ymax - ymin

                if width >= min_width and height >= min_height:
                    person_detections.append((xmin, ymin, xmax, ymax))

        print(f"Persons detected: {len(person_detections)}")
        return person_detections

    def draw_bounding_boxes(self, frame, tracks, tracked_faces, face_flag, image_name):
        print("Drawing bounding boxes...")
        for track_id, bbox in self.previous_tracks.items():
            if track_id in tracked_faces:
                if face_flag == 'true':
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    cv2.putText(frame, f"person_ID : {image_name}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    face_ids = tracked_faces[track_id]
                    person_id = face_ids.split('_')[1]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    cv2.putText(frame, f"person_ID : {person_id}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for track_id, bbox in self.previous_tracks.items():
            if track_id not in tracked_faces:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, f"track_ID : {str(track_id)}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def recognize_faces(self, frame, frame_number, output_dir, known_faces, tracker, video_name, yolo_model, gender_model, age_model, face_flag, image_name):
        if frame is None:
            print(f"Warning: Frame {frame_number} is None.")
            return frame

        print(f"Processing frame {frame_number}...")
        original_shape = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detect_faces = self.extract_embeddings(frame_rgb)
        person_detections = self.detect_persons(frame, yolo_model)

        results = []
        for (xmin, ymin, xmax, ymax) in person_detections:
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], 1.0, 0])

        tracks = tracker.update_tracks(results, frame=frame)
        print(f"Tracks updated: {len(tracks)}")

        for track_id in list(self.previous_tracks.keys()):
            bbox = self.previous_tracks[track_id]
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > frame.shape[1] or bbox[3] > frame.shape[0]:
                del self.previous_tracks[track_id]

        active_track_ids = {track.track_id for track in tracks if track.is_confirmed()}
        inactive_track_ids = set(self.tracked_faces.keys()) - active_track_ids

        for inactive_track_id in inactive_track_ids:
            if self.tracked_faces[inactive_track_id] in self.known_faces:
                face_id = self.tracked_faces[inactive_track_id]
                embedding = self.known_faces[face_id][1]
                self.inactive_faces[face_id] = embedding
                self.assigned_ids.discard(face_id)
                del self.known_faces[face_id]
            del self.tracked_faces[inactive_track_id]

        for track_id in list(self.previous_tracks.keys()):
            if track_id not in active_track_ids:
                del self.previous_tracks[track_id]

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            track_bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))

            cv2.rectangle(frame, (track_bbox[0], track_bbox[1]), (track_bbox[2], track_bbox[3]), (255, 0, 0), 2)

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
                    if track_id in self.tracked_faces:
                        face_id = self.tracked_faces[track_id]
                        print(f"Track ID {track_id} already assigned to face ID {face_id}.")
                    else:
                        face_id = self.assign_face_id(embedding, known_faces)
                        if face_id is not None and face_id not in self.assigned_ids:
                            self.tracked_faces[track_id] = face_id
                            self.known_faces[face_id] = (track_id, embedding)
                            self.assigned_ids.add(face_id)
                            print(f"Track ID {track_id} assigned to new face ID {face_id}.")
                        elif face_id in self.assigned_ids:
                            continue
                        else:
                            max_similarity = 0
                            best_face_id = None
                            for known_face_id, (known_track_id, known_embedding) in self.known_faces.items():
                                similarity = self.compare_similarity(embedding, known_embedding)
                                if similarity > max_similarity:
                                    max_similarity = similarity
                                    best_face_id = known_face_id
                            if max_similarity > 0.8 and best_face_id not in self.assigned_ids:
                                self.tracked_faces[track_id] = best_face_id
                                self.known_faces[best_face_id] = (track_id, embedding)
                                self.assigned_ids.add(best_face_id)
                                print(f"Reassigned face ID {best_face_id} to track {track_id}.")
                    break

            self.previous_tracks[track_id] = track_bbox

            face_image = frame[track_bbox[1]:track_bbox[3], track_bbox[0]:track_bbox[2]]
            if face_image is None or face_image.size == 0:
                print(f"Warning: Extracted face image is None or empty for track {track_id}.")
                continue

        self.draw_bounding_boxes(frame, tracks, self.tracked_faces, face_flag, image_name)
        print(f"Frame {frame_number} processed.")

        return frame

def calculate_tracking_parameters(interval):
    base_max_age = 15
    base_nn_budget = 15
    
    max_age = base_max_age + int(interval / 10)
    nn_budget = base_nn_budget + int(interval / 10)
    
    return max_age, nn_budget

def predict_gender(face_image, gender_model):
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    results = gender_model.predict(source=[face_rgb], save=False)
    genders = {0: "Male", 1: "Female"}
    gender_id = results[0].boxes.data[0][5].item()
    return genders.get(gender_id, "Unknown")

# H.264 코덱으로 재인코딩하는 함수
def reencode_video(input_path, output_path):
    command = [
        'ffmpeg', '-i', input_path, '-c:v', 'libx264', '-b:v', '2000k',
        '-c:a', 'aac', '-b:a', '128k', output_path
    ]
    subprocess.run(command, check=True)

def process_video(video_path, output_dir, known_face_paths, yolo_model_path, gender_model_path, age_model_path, person_id, face_flag, image_name, interval=3, target_fps=1):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    recognizer = FaceRecognizer()

    v_cap = cv2.VideoCapture(video_path)
    if not v_cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_rate = int(v_cap.get(cv2.CAP_PROP_FPS))
    frame_width, frame_height = None, None
    video_writer = None
    print(f"loading face : {known_face_paths}")
    known_faces = recognizer.load_known_faces(known_face_paths)
    yolo_model = YOLO(yolo_model_path)
    gender_model = YOLO(gender_model_path)
    age_model = ResNetAgeModel(num_classes=4)
    age_model.load_state_dict(torch.load(age_model_path))
    age_model = age_model.to(device)
    age_model.eval()

    max_age, nn_budget = calculate_tracking_parameters(interval)
    tracker = DeepSort(max_age=max_age, n_init=2, nn_budget=nn_budget)

    frame_number = 0
    temp_video_path = os.path.join(output_dir, f"{video_name}_clip", f"{video_name}_temp_output.mp4")

    while True:
        success, frame = v_cap.read()
        if not success or frame is None:
            print(f"Failed to read frame {frame_number} from {video_path}")
            break
        frame_number += 1

        if frame_number % interval == 0:
            frame = recognizer.recognize_faces(frame, frame_number, output_dir, known_faces, tracker, video_name, yolo_model, gender_model, age_model, face_flag, image_name)
        else:
            recognizer.draw_bounding_boxes(frame, [], recognizer.tracked_faces, face_flag, image_name)
        
        if frame_width is None or frame_height is None:
            frame_height, frame_width = frame.shape[:2]
            os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)
            video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

        if video_writer:
            video_writer.write(frame)

    v_cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    # Reencode the video using H.264 codec
    output_video_path = os.path.join(output_dir, f"{video_name}_clip", f"{video_name}_output.mp4")
    reencode_video(temp_video_path, output_video_path)

    # Remove the temporary file
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

def process_videos(video_directory, output_dir, known_face_paths, yolo_model_path, gender_model_path, age_model_path, interval, face_flag, image_name, video_name=None, target_fps=10):
    video_paths = [os.path.join(video_directory, file) for file in os.listdir(video_directory) if file.endswith(('.mp4', '.avi', '.mov'))]
    
    if video_name:
        video_paths = [video_path for video_path in video_paths if os.path.splitext(os.path.basename(video_path))[0] == video_name]

    target_recognizer = FaceRecognizer()
    target_faces = target_recognizer.load_known_faces(known_face_paths)
    if not target_faces:
        print("No known faces found.")
        return
    
    target_embeddings = [face['embedding'] for face in target_faces]
    all_known_face_paths = []
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        clip_folder = os.path.join(output_dir, f"{video_name}_clip")
        if not os.path.exists(clip_folder):
            continue
        person_folders = [d for d in os.listdir(clip_folder) if os.path.isdir(os.path.join(clip_folder, d))]

        for person_id in person_folders:
            person_folder_path = os.path.join(clip_folder, person_id)
            known_face_paths = [
                os.path.join(person_folder_path, img_file)
                for img_file in os.listdir(person_folder_path)
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            all_known_face_paths.extend(known_face_paths)

        print(all_known_face_paths)
        known_faces = target_recognizer.load_known_faces(all_known_face_paths)
        print(known_faces)
        if not known_faces:
            print("No embeddings found for any faces.")
            continue

        matched_faces = []
        for face in known_faces:
            similarities = []
            for target_embedding in target_embeddings:
                similarity = target_recognizer.compare_similarity(target_embedding, face['embedding'])
                similarities.append(similarity)
            
            if any(sim >= 0.64 for sim in similarities):
                matched_faces.append(face['id'])

        if matched_faces:
            print(f"Matched faces: {matched_faces}")
            combined_face_ids = ','.join(matched_faces)
            process_video(video_path, output_dir, all_known_face_paths, yolo_model_path, gender_model_path, age_model_path, combined_face_ids, face_flag, image_name, interval, target_fps)
            break

if __name__ == "__main__":
    try:
        video_name = sys.argv[1]
        user_id = sys.argv[2]
        filter_id = sys.argv[3]
        known_face_paths_str = sys.argv[4]
        image_name = sys.argv[5]
        face_flag = sys.argv[6].lower()

        known_face_paths = known_face_paths_str.split(',')
        video_directory = f"./uploaded_videos/{user_id}/"

        output_directory = f"./extracted_images/{user_id}/filter_{filter_id}"
        yolo_model_path = './models/yolov8x.pt'
        gender_model_path = './models/gender_model.pt'
        age_model_path = './models/age_best.pth'

        interval = 3
        process_videos(video_directory, output_directory, known_face_paths, yolo_model_path, gender_model_path, age_model_path, interval, face_flag, image_name, video_name=video_name)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
