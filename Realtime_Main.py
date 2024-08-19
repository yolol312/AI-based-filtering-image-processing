from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os, base64
import pymysql
import subprocess
import numpy as np
import cv2
from datetime import datetime
from filelock import FileLock
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading
from collections import defaultdict, Counter
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import json
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)

# Activation Function
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Convolutional Layer
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Bottleneck Layer
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        hidden_channels = out_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.cv2 = Conv(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))

# C2f Layer
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

# SPPF Layer
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

# Detect Layer for Single Class and Single Scale Prediction
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

# YOLOv8x Model for Single Class and Single Scale Prediction
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

# 각 웹캠의 이미지가 저장될 폴더 경로 설정
SAVE_FOLDER = 'realtime_saved_images'
os.makedirs(SAVE_FOLDER, exist_ok=True)

# 처리 시작 시간
starttime = ""

#예측 중인지 확인하는 변수
process_playing = False

# 커스텀 모델 불러오기
def load_model(model_path, device):
    model = YOLOvBIT(num_classes=1)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# YOLO 모델 및 DeepSort 초기화
yolo_model = YOLO('models/yolov8x.pt')  # YOLO 모델 경로를 적절히 설정하세요
tracker = DeepSort(max_age=30, nn_budget=20)  # DeepSort 트래커 설정

def detect_persons(frame, yolo_model):
    print("Detecting persons...")
    yolo_results = yolo_model.predict(source=[frame], save=False)[0]
    person_detections = [
        (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
        for data in yolo_results.boxes.data.tolist()
        if float(data[4]) >= 0.85 and int(data[5]) == 0
    ]
    print(f"Persons detected: {len(person_detections)}")
    return person_detections

# folder_path에 있는 이미지 파일 수를 계산하는 함수
def count_images_in_folder(folder_path):
    # 이미지 파일 확장자 리스트
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    
    # 폴더 내 이미지 파일 수 계산
    image_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)])
    
    return image_count

def get_cameras_by_user_no(user_no):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT cam_num, cam_name FROM realtime_camera WHERE user_no = %s"
            cursor.execute(sql, (user_no,))
            cameras = cursor.fetchall()
            return cameras
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

def get_origin_videos_by_cam_num(cam_num):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT or_video_id, or_video_name
                FROM realtime_origin_video
                WHERE cam_num = %s
            """
            cursor.execute(sql, (cam_num,))
            origin_videos = cursor.fetchall()
            return origin_videos
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

def get_log_by_or_video_id(or_video_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT log_data
                FROM realtime_log
                WHERE or_video_id = %s
            """
            cursor.execute(sql, (or_video_id,))
            log_datas = cursor.fetchall()
            return log_datas
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

def process_log_data(log_datas):
    # 로그 데이터를 파싱하여 Track_ID와 나머지 정보를 추출합니다.
    track_id_details = defaultdict(lambda: {
        'person_id': [],
        'image': [],
        'gender': [],
        'age': [],
        'upclothes': [],
        'downclothes': [],
    })
    
    for log in log_datas:
        log = log['log_data']  # log_datas가 딕셔너리 리스트라고 가정
        matches = re.match(r"Track ID: (\d+), Person ID: (\d+), Image: (.+?), Gender: (.+?), Age: (.+?), UpClothes: (.+?), DownClothes: (.+?)$", log)
        if matches:
            track_id = matches.group(1)
            track_id_details[track_id]['person_id'].append(matches.group(2))
            track_id_details[track_id]['image'].append(matches.group(3))
            track_id_details[track_id]['gender'].append(matches.group(4))
            track_id_details[track_id]['age'].append(matches.group(5))
            track_id_details[track_id]['upclothes'].append(matches.group(6))
            track_id_details[track_id]['downclothes'].append(matches.group(7))
    
    # 각 Track_ID에 대해 가장 자주 나타나는 정보를 찾습니다.
    result = {}
    for track_id, details in track_id_details.items():
        most_common_person_id = get_most_common_value(details['person_id'])
        most_common_image = get_most_common_value(details['image'])
        most_common_gender = get_most_common_value(details['gender'])
        most_common_age = get_most_common_value(details['age'])
        most_common_upclothes = get_most_common_value(details['upclothes'])
        most_common_downclothes = get_most_common_value(details['downclothes'])

        result[track_id] = {
            "Track ID": track_id,
            "Gender": most_common_gender,
            "Age": most_common_age,
            "UpClothes": most_common_upclothes,
            "DownClothes": most_common_downclothes
        }
    
    return result

def update_cam_name(cam_num, new_cam_name):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "UPDATE realtime_camera SET cam_name = %s WHERE cam_num = %s"
            cursor.execute(sql, (new_cam_name, cam_num))
            connection.commit()
            return True
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        connection.rollback()  # 롤백 처리
        return False
    finally:
        connection.close()

def get_pro_video_content(or_video_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT pro_video_content
                FROM realtime_processed_video
                WHERE or_video_id = %s
            """
            cursor.execute(sql, (or_video_id,))
            result = cursor.fetchone()
            return result['pro_video_content'] if result else None
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

def delete_analysis_results(or_video_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # 트랜잭션 시작
            connection.begin()

            # 로그 데이터 삭제
            sql = "DELETE FROM realtime_log WHERE or_video_id = %s"
            cursor.execute(sql, (or_video_id,))

            # 처리된 비디오 데이터 삭제
            sql = "DELETE FROM realtime_processed_video WHERE or_video_id = %s"
            cursor.execute(sql, (or_video_id,))

            # 원본 비디오 데이터 삭제
            sql = "DELETE FROM realtime_origin_video WHERE or_video_id = %s"
            cursor.execute(sql, (or_video_id,))

            # 트랜잭션 커밋
            connection.commit()
            return True
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        connection.rollback()  # 트랜잭션 롤백
        return False
    finally:
        connection.close()

def delete_videos(cam_num):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # 트랜잭션 시작
            connection.begin()

            # cam_num에 해당하는 모든 or_video_id를 가져옴
            sql = "SELECT or_video_id FROM realtime_origin_video WHERE cam_num = %s"
            cursor.execute(sql, (cam_num,))
            video_ids = cursor.fetchall()

            for video in video_ids:
                or_video_id = video['or_video_id']
                
                # 로그 데이터 삭제
                sql = "DELETE FROM realtime_log WHERE or_video_id = %s"
                cursor.execute(sql, (or_video_id,))

                # 처리된 비디오 데이터 삭제
                sql = "DELETE FROM realtime_processed_video WHERE or_video_id = %s"
                cursor.execute(sql, (or_video_id,))

                # 원본 비디오 데이터 삭제
                sql = "DELETE FROM realtime_origin_video WHERE or_video_id = %s"
                cursor.execute(sql, (or_video_id,))

            # 트랜잭션 커밋
            connection.commit()
            return True
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        connection.rollback()  # 트랜잭션 롤백
        return False
    finally:
        connection.close()

# ffmpeg를 사용하여 비디오를 H.264 코덱으로 재인코딩.
def reencode_video(input_path, output_path):
    command = [
        'ffmpeg', '-i', input_path, '-c:v', 'libx264', '-b:v', '2000k',
        '-c:a', 'aac', '-b:a', '128k', output_path
    ]
    subprocess.run(command, check=True)

def get_user_no(user_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT user_no FROM user WHERE user_id = %s"
            cursor.execute(sql, (user_id,))
            result = cursor.fetchone()
            if result:
                return result['user_no']
            else:
                print(f"No record found for user_id: {user_id}")
                return None
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

# user_no로 cam_num 정보 가져오기
def get_cam_num(user_no):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT cam_num FROM realtime_camera WHERE user_no = %s"
            cursor.execute(sql, (user_no))
            result = cursor.fetchone()
            if result:
                return result['cam_num']
            else:
                print(f"No record found for user_no: {user_no}")
                return None
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

# user_no로 cam_name 정보 가져오기
def get_cam_name(user_no):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT cam_name FROM realtime_camera WHERE user_no = %s"
            cursor.execute(sql, (user_no))
            result = cursor.fetchone()
            if result:
                return result['cam_name']
            else:
                print(f"No record found for user_no: {user_no}")
                return None
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

# 파일명에서 시작 시간 추출하기
def convert_filename_to_datetime(filename):
    # 파일명에서 날짜와 시간을 추출
    datetime_str = filename.split('_')[0] + filename.split('_')[1]

    # 추출된 문자열을 datetime 객체로 변환
    datetime_obj = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")

    # 원하는 형식의 문자열로 변환
    formatted_datetime = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")

    return formatted_datetime

# cropped_images_path에 있는 모든 이미지 파일을 삭제하는 함수
def remove_image_files_in_directory(directory_path):
    # 디렉토리가 존재하는지 확인
    if os.path.exists(directory_path):
        # 디렉토리 내 모든 이미지 파일 삭제
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                os.remove(file_path)
        print(f"All image files in {directory_path} have been removed.")
    else:
        print(f"The directory {directory_path} does not exist.")

def rename_video_name(video_name):
    # 언더스코어로 분리하여 날짜 및 시간 부분 추출
    parts = video_name.split("_")
    prefix = parts[0]  # "Woosong"
    date_part = parts[1]  # "20240819"
    time_part = parts[2]  # "143608"
    suffix1 = parts[3]  # "original"
    suffix2 = parts[4]  # "video.mp4"

    # 날짜와 시간을 datetime 객체로 변환
    datetime_obj = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")

    # 원하는 형식으로 변환
    formatted_date_time = datetime_obj.strftime("%Y-%m-%d_%H:%M:%S")

    # 새로운 파일명 생성
    new_filename = f"{prefix}_{formatted_date_time}_{suffix1}_{suffix2}"

    return new_filename

# 영상 및 log 결과 DB에 저장
def save_video_to_db(user_no, cam_num, start_time, origin_video_path, processed_video_path, output_txt_path, processed_log_txt_path, cropped_images_path):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # 원본 비디오를 origin_video 테이블에 저장
            origin_video_name = os.path.basename(origin_video_path)
            origin_video_rename = rename_video_name(origin_video_name)
            insert_origin_video_query = """
                INSERT INTO realtime_origin_video (or_video_name, or_video_content, start_time, cam_num)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_origin_video_query, (origin_video_rename, origin_video_path, start_time, cam_num))
            or_video_id = cursor.lastrowid  # 방금 삽입된 원본 비디오의 ID

            # 처리된 비디오를 processed_video 테이블에 저장
            processed_video_name = os.path.basename(processed_video_path)
            processed_video_rename = rename_video_name(processed_video_name)
            insert_processed_video_query = """
                INSERT INTO realtime_processed_video (pro_video_name, pro_video_content, or_video_id)
                VALUES (%s, %s, %s)
            """
            cursor.execute(insert_processed_video_query, (processed_video_rename, processed_video_path, or_video_id))

            # txt 파일 내용을 log 테이블에 저장
            with open(output_txt_path, 'r') as txt_file:
                for line in txt_file:
                    log_data = line.strip()
                    insert_log_query = """
                        INSERT INTO realtime_log (log_data, or_video_id)
                        VALUES (%s, %s)
                    """
                    cursor.execute(insert_log_query, (log_data, or_video_id))

            # 변경 사항 커밋
            connection.commit()

            print("Videos and logs have been saved to the database.")

            while process_playing :
                continue

            os.remove(output_txt_path)
            os.remove(processed_log_txt_path)
            remove_image_files_in_directory(cropped_images_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        connection.rollback()
    finally:
        connection.close()

def get_most_common_value(values):
    counter = Counter(values)
    most_common = counter.most_common()
    for value, _ in most_common:
        if value != "unknown":
            return value
    return "unknown"

def summarize_tracking_data(txt_file_path, cropped_images_dir):
    if not os.path.exists(txt_file_path):
        return "Making summary predictions...", {}
    
    tracking_data = defaultdict(lambda: {
        'gender': [],
        'age': [],
        'upclothes': [],
        'downclothes': [],
    })

    with open(txt_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(', ')
            track_id = None
            gender = None
            age = None
            upclothes = None
            downclothes = None
            
            for part in parts:
                if part.startswith("Track ID:"):
                    track_id = part.split(": ")[1]
                elif part.startswith("Gender:"):
                    gender = part.split(": ")[1]
                elif part.startswith("Age:"):
                    age = part.split(": ")[1]
                elif part.startswith("UpClothes:"):
                    upclothes = part.split(": ")[1]
                elif part.startswith("DownClothes:"):
                    downclothes = part.split(": ")[1]

            if track_id is not None:
                tracking_data[track_id]['gender'].append(gender)
                tracking_data[track_id]['age'].append(age)
                tracking_data[track_id]['upclothes'].append(upclothes)
                tracking_data[track_id]['downclothes'].append(downclothes)

    summary = {}
    person_image_base64 = {}

    for track_id, data in tracking_data.items():
        most_common_gender = get_most_common_value(data['gender'])
        most_common_age = get_most_common_value(data['age'])
        most_common_upclothes = get_most_common_value(data['upclothes'])
        most_common_downclothes = get_most_common_value(data['downclothes'])

        # 딕셔너리 형태로 Track ID별 요약 정보를 저장
        summary[track_id] = {
            "Gender": most_common_gender,
            "Age": most_common_age,
            "Upclothes": most_common_upclothes,
            "Downclothes": most_common_downclothes
        }

        image_filename = f"Person_{track_id}.jpg"
        image_path = os.path.join(cropped_images_dir, image_filename)
        if os.path.exists(image_path):
            person_image = cv2.imread(image_path)
            _, buffer = cv2.imencode('.jpg', person_image)
            person_image_base64[track_id] = base64.b64encode(buffer).decode('utf-8')

    return summary, person_image_base64

# 텍스트 파일의 마지막 줄을 읽어서 반환하는 함수
def get_last_line_from_txt(txt_file):
    if not os.path.exists(txt_file):
        return "Predicting..."

    with open(txt_file, 'rb') as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:  # 파일이 비어 있는 경우
            return "Predicting..."

        # 파일이 비어 있지 않은 경우
        f.seek(-1, os.SEEK_END)
        while f.tell() > 0:
            f.seek(-2, os.SEEK_CUR)
            if f.read(1) == b'\n':
                break

        last_line = f.readline().decode()
    
    return last_line.strip()

# 데이터베이스 연결 함수
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='1234',
        database='wb39_project_realtime',
        cursorclass=pymysql.cursors.DictCursor
    )

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

# 백그라운드에서 실행할 함수 정의
def run_background_process(user_id, user_cam_folder_path, processed_folder_path):
    global process_playing
    process = subprocess.Popen(
        ["python", "Realtime_Prediction_YOLOL.py", user_id, user_cam_folder_path, processed_folder_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    process_playing = False

    if process.returncode != 0:
        print(f"Error occurred: {stderr.decode()}")

# 1. 회원가입 (Post)
@app.route('/realtime_signup', methods=['POST'])
def realtime_signup():
    data = request.get_json()
    print("Received data:", data)  # 디버깅 메시지 추가
    if data and isinstance(data, list):
        connection = get_db_connection()
        cursor = connection.cursor()
        
        try:
            for user in data:
                print("Processing user:", user)  # 디버깅 메시지 추가
                # 아이디 중복 체크
                check_sql = "SELECT * FROM user WHERE user_id = %s"
                cursor.execute(check_sql, (user['user_id'],))
                result = cursor.fetchone()
                
                print("Check result:", result)  # 디버깅 메시지 추가
                if result is not None and result['user_id'] == user['user_id']:
                    print(f"User ID {user['user_id']} already exists")  # 디버깅 메시지 추가
                    return jsonify({"error": "이미 존재하는 ID입니다"}), 409

                # User 테이블에 삽입
                user_sql = "INSERT INTO user (user_id) VALUES (%s)"
                cursor.execute(user_sql, (user['user_id'],))
                print("Inserted into user table")  # 디버깅 메시지 추가

                # 방금 삽입한 user_no 가져오기
                user_no = cursor.lastrowid
                print("Inserted user_no:", user_no)  # 디버깅 메시지 추가

                # Password 테이블에 삽입
                password_sql = "INSERT INTO realtime_camera (cam_name, user_no) VALUES (%s, %s)"
                cursor.execute(password_sql, (user['cam_name'], user_no))
                print("Inserted into password table")  # 디버깅 메시지 추가

            connection.commit()
            print("Transaction committed")  # 디버깅 메시지 추가
        except KeyError as e:
            connection.rollback()
            print(f"KeyError: {str(e)}")  # 디버깅 메시지 추가
            return jsonify({"error": f"Invalid data format: missing {str(e)}"}), 400
        except Exception as e:
            connection.rollback()
            print(f"Exception: {str(e)}")  # 디버깅 메시지 추가
            return jsonify({"error": str(e)}), 500
        finally:
            cursor.close()
            connection.close()
            print("Connection closed")  # 디버깅 메시지 추가
        
        return jsonify({"message": "Data received and stored successfully"}), 200
    else:
        print("No data received or invalid format")  # 디버깅 메시지 추가
        return jsonify({"error": "No data received or invalid format"}), 400

# 2. 로그인 (Post)
@app.route('/realtime_login', methods=['POST'])
def realtime_login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data received"}), 400
        if isinstance(data, str):
            return jsonify({"status": "error", "message": "Invalid JSON data format"}), 400
        
        user_id = data.get('user_id', '')

        user_no = get_user_no(user_id)
        cam_name = get_cam_name(user_no)

        return jsonify({"status": "success", "message": "User and Camera added successfully", "cam_name": cam_name}), 200

    except ValueError as e:
        print(f"A ValueError occurred: {str(e)}")
        return jsonify({"status": "error", "message": f"A ValueError occurred: {str(e)}"}), 400
    except KeyError as e:
        print(f"A KeyError occurred: {str(e)}")
        return jsonify({"status": "error", "message": f"A KeyError occurred: {str(e)}"}), 400
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500

# 3. 캠 이름 수정 (Post)
@app.route('/realtime_cam_update', methods=['POST'])
def realtime_cam_update():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data received"}), 400
        if isinstance(data, str):
            return jsonify({"status": "error", "message": "Invalid JSON data format"}), 400
        
        data = data[0]
        user_id = data.get('user_id', '')
        new_cam_name = data.get('cam_name', '')

        user_no = get_user_no(user_id)
        cam_num = get_cam_num(user_no)

        if not cam_num:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
        if not new_cam_name:
            return jsonify({"status": "error", "message": "New camera name not provided"}), 400

        # or_video_id에 해당하는 데이터 삭제
        if not delete_videos(cam_num):
            return jsonify({"status": "error", "message": "Error deleting data from the database"}), 500

        if update_cam_name(cam_num, new_cam_name):
            return jsonify({"status": "success", "message": "Camera name updated successfully", "new_cam_name": new_cam_name}), 200
        else:
            return jsonify({"status": "error", "message": "Failed to update camera name"}), 500

    except ValueError as e:
        print(f"A ValueError occurred: {str(e)}")
        return jsonify({"status": "error", "message": f"A ValueError occurred: {str(e)}"}), 400
    except KeyError as e:
        print(f"A KeyError occurred: {str(e)}")
        return jsonify({"status": "error", "message": f"A KeyError occurred: {str(e)}"}), 400
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500

# 4. 실시간 웹캠 이미지 전송 (Post)
@app.route('/realtime_upload_image', methods=['POST'])
def realtime_upload_image():
    # 'json_data'라는 form 필드에서 JSON 문자열을 가져옴
    json_data = request.form.get('json_data')
    
    if not json_data:
        return jsonify({"status": "error", "message": "No JSON data received"}), 400

    # JSON 문자열을 Python 딕셔너리로 변환
    data = json.loads(json_data)

    # 'user_data' 딕셔너리에서 'user_id'와 'cam_name'을 추출
    user_data = data.get('user_data', {})
    user_id = user_data.get('user_id', '')

    user_no = get_user_no(user_id)
    cam_name = get_cam_name(user_no)

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    if not cam_name:
        return jsonify({"error": "Cam Name is required"}), 400

    # 이미지 파일 수신
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Image decoding failed"}), 500

    # 결과 이미지를 Base64로 인코딩하여 클라이언트에 반환
    _, buffer = cv2.imencode('.jpg', img)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # 이미지 파일 저장
    starttime = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{starttime}_{file.filename}.jpg"
    user_folder_path = os.path.join(SAVE_FOLDER, user_id)
    user_cam_folder_path = os.path.join(user_folder_path, cam_name)
    origin_folder_path = os.path.join(user_cam_folder_path, "origin_images")
    processed_folder_path = os.path.join(user_cam_folder_path, "processed_images")
    cropped_images_path = os.path.join(user_cam_folder_path, "cropped_images").replace("\\", "/")
    output_txt_path = os.path.join(user_cam_folder_path, "predictions.txt").replace("\\", "/")

    origin_filepath = os.path.join(origin_folder_path, filename).replace("\\", "/")
    processed_filepath = os.path.join(processed_folder_path, filename).replace("\\", "/")

    # 디렉토리가 존재하지 않을 경우 생성
    os.makedirs(user_folder_path, exist_ok=True)
    os.makedirs(user_cam_folder_path, exist_ok=True)
    os.makedirs(origin_folder_path, exist_ok=True)
    os.makedirs(processed_folder_path, exist_ok=True)
    os.makedirs(cropped_images_path, exist_ok=True)

    cv2.imwrite(origin_filepath, img)

    # 이미지 파일 수 계산
    image_count = count_images_in_folder(origin_folder_path)

    print(f"Current count value: {image_count}")

    global process_playing, yolo_model, device

    if process_playing == False:
        print("Threading Start")

        # 새로운 스레드를 생성하여 백그라운드에서 스크립트를 실행
        threading.Thread(
            target=run_background_process,
            args=(user_id, user_cam_folder_path, processed_folder_path)
        ).start()
        process_playing = True


    if image_count % 5 == 0 and image_count % 150 != 0:
        # 실시간 트래킹 처리
        person_detections = detect_persons(img, yolo_model)
        
        results = []
        previous_tracks = {}
        for (xmin, ymin, xmax, ymax) in person_detections:
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], 1.0, 0])

        tracker_outputs = tracker.update_tracks(results, frame=img)

        active_track_ids = {track.track_id for track in tracker_outputs if track.is_confirmed()}

        for track_id in list(previous_tracks.keys()):
            if track_id not in active_track_ids:
                del previous_tracks[track_id]

        # 활성 트랙 갱신 및 트랙 상태 시각화
        for track in tracker_outputs:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            track_bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))

            previous_tracks[track_id] = track_bbox

            # 트래킹 결과를 이미지에 시각화
            cv2.rectangle(img, (int(track_bbox[0]), int(track_bbox[1])), (int(track_bbox[2]), int(track_bbox[3])), (255, 0, 0), 2)
            cv2.putText(img, f"Person ID: {track_id}", (int(track_bbox[0]), int(track_bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 결과 이미지를 다시 저장
        cv2.imwrite(processed_filepath, img)

        # 결과 이미지를 Base64로 인코딩하여 클라이언트에 반환
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        predictions = get_last_line_from_txt(output_txt_path)

        print(f"Received and processed image from cam {cam_name} with shape: {img.shape} as {filename} / predictions : {predictions}")
        return jsonify({"message": "Image received and processed", "frame": image_base64, "person_data" : predictions}), 200
    '''
    if image_count % 6 == 0 and image_count % 150 != 0:
        print("Process Start")
        # 실시간 트래킹 처리
        person_detections = detect_objects(yolo_model, img, device)
        
        results = []
        for (xmin, ymin, xmax, ymax) in person_detections:
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], 1.0, 0])

        tracker_outputs = tracker.update_tracks(results, frame=img)

        # 트래킹 결과를 이미지에 시각화
        for track in tracker_outputs:
            bbox = track.to_tlbr()
            track_id = track.track_id
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(img, f"Person ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 결과 이미지를 다시 저장
        cv2.imwrite(processed_filepath, img)

        # 결과 이미지를 Base64로 인코딩하여 클라이언트에 반환
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        predictions = get_last_line_from_txt(output_txt_path)

        print(f"Received and processed image from cam {cam_name} with shape: {img.shape} as {filename} / predictions : {predictions}")
        return jsonify({"message": "Image received and processed", "frame": image_base64, "person_data" : predictions}), 200
    '''
    if image_count % 150 == 0 and image_count != 0:
        # 결과 이미지를 Base64로 인코딩하여 클라이언트에 반환
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        summary_predictions, person_images = summarize_tracking_data(output_txt_path, cropped_images_path)

        print(f"Received and processed image from cam {cam_name} with shape: {img.shape} as {filename} / Summary Predictions : {summary_predictions} / person_image : {person_images}")
        return jsonify({"message": "Image received and processed", "frame": image_base64, "person_data": summary_predictions, "person_image" : person_images}), 200

    # 결과 이미지를 Base64로 인코딩하여 클라이언트에 반환
    _, buffer = cv2.imencode('.jpg', img)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    predictions = get_last_line_from_txt(output_txt_path)

    print(f"Received and return image from cam {cam_name} with shape: {img.shape} as {filename} / predictions : {predictions}")
    return jsonify({"message": "Image received and processed", "person_data" : predictions}), 200

# 5. 실시간 웹캠 이미지 전송 종료 (Post)
@app.route('/realtime_upload_image_end', methods=['POST'])
def realtime_upload_image_end():
    data = request.get_json()

    # JSON 데이터가 제대로 수신되지 않았을 경우 확인
    if not data:
        return jsonify({"status": "error", "message": "No JSON data received"}), 400

    # 수신된 데이터가 문자열이 아닌 JSON 객체인지 확인
    if isinstance(data, str):
        return jsonify({"status": "error", "message": "Invalid JSON data format"}), 400

    user_data = data.get('user_data', {})
    user_id = user_data.get('user_id', '')
    
    user_no = get_user_no(user_id)
    cam_name = get_cam_name(user_no)
    cam_num = get_cam_num(user_no)

    # Processed 이미지 폴더 경로 설정
    user_folder_path = os.path.join(SAVE_FOLDER, user_id)
    user_cam_folder_path = os.path.join(user_folder_path, cam_name)
    save_origin_path = os.path.join(user_cam_folder_path, "origin_video")
    save_processed_path = os.path.join(user_cam_folder_path, "processed_video")
    origin_folder_path = os.path.join(user_cam_folder_path, "origin_images")
    processed_folder_path = os.path.join(user_cam_folder_path, "processed_images")
    cropped_images_path = os.path.join(user_cam_folder_path, "cropped_images")
    output_txt_path = os.path.join(user_cam_folder_path, "predictions.txt").replace("\\", "/")
    processed_log_txt_path = os.path.join(user_cam_folder_path, "processed_images.log").replace("\\", "/")
    
    os.makedirs(save_origin_path, exist_ok=True)
    os.makedirs(save_processed_path, exist_ok=True)

    # 이미지 파일 목록 가져오기
    pro_image_files = sorted([f for f in os.listdir(processed_folder_path) if f.endswith('.jpg')])
    or_image_files = sorted([f for f in os.listdir(origin_folder_path) if f.endswith('.jpg')])

    # 이미지가 있는지 확인
    if len(or_image_files) == 0:
        return jsonify({"status": "error", "message": "No images found to create video"}), 404
    
    if len(pro_image_files) == 0:
        return jsonify({"status": "error", "message": "No images found to create video"}), 404
    

    # 언더스코어(_)로 분리
    parts = or_image_files[0].split("_")

    # 첫 번째와 두 번째 부분을 합침
    start_time = f"{parts[0]}_{parts[1]}"

    # 첫 번째 이미지를 읽어 크기를 가져옴
    first_image_path = os.path.join(origin_folder_path, or_image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # 원본 비디오 작성 준비
    origin_video_path = os.path.join(save_origin_path, f"{cam_name}_{start_time}_original_video.mp4")
    origin_video_writer = cv2.VideoWriter(origin_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    print(f"Origin Video Writer opened: {origin_video_writer.isOpened()}")

    # 원본 이미지를 사용하여 원본 비디오 생성
    for image_file in or_image_files:
        image_path = os.path.join(origin_folder_path, image_file)
        image = cv2.imread(image_path)
        origin_video_writer.write(image)

    origin_video_writer.release()

    # 원본 비디오 작성 완료
    print(f"Original video created at {origin_video_path}")

    # 합쳐진 비디오 작성 준비
    processed_video_path = os.path.join(save_processed_path, f"{cam_name}_{start_time}_processed_video.mp4")
    processed_video_writer = cv2.VideoWriter(processed_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    print(f"Processed Video Writer opened: {processed_video_writer.isOpened()}")

    # 이미지들을 합쳐진 비디오에 작성 (원본 이미지와 처리된 이미지를 비교)
    for image_file in pro_image_files:
        image_path = os.path.join(processed_folder_path, image_file)
        image = cv2.imread(image_path)
        processed_video_writer.write(image)

    processed_video_writer.release()

    # 합쳐진 비디오 작성 완료
    print(f"Combined video created at {processed_video_path}")
    
    start_time = convert_filename_to_datetime(or_image_files[0])

    save_video_to_db(user_no, cam_num, start_time, origin_video_path, processed_video_path, output_txt_path, processed_log_txt_path, cropped_images_path)
    
    # 이미지 삭제(처리된 이미지)
    for image_file in pro_image_files:
        image_path = os.path.join(processed_folder_path, image_file)
        os.remove(image_path)

    # 이미지 삭제(원본 이미지)
    for image_file in or_image_files:
        image_path = os.path.join(origin_folder_path, image_file)
        os.remove(image_path)

    print(f"Received and saved image from cam {cam_name} End")

    return jsonify({"message": "Image received and saved"}), 200

# 6. 이전 분석 불러오기 (Post)
@app.route('/realtime_search_previous_analysis', methods=['POST'])
def realtime_search_previous_analysis():
    data = request.get_json()

    # JSON 데이터가 제대로 수신되지 않았을 경우 확인
    if not data:
        return jsonify({"status": "error", "message": "No JSON data received"}), 400

    # 수신된 데이터가 문자열이 아닌 JSON 객체인지 확인
    if isinstance(data, str):
        return jsonify({"status": "error", "message": "Invalid JSON data format"}), 400

    user_data = data.get('user_data', {})
    user_id = user_data.get('user_id', '')
    
    user_no = get_user_no(user_id)
    
    # user_no에 해당하는 모든 카메라를 가져옴
    cameras = get_cameras_by_user_no(user_no)
    
    if cameras is None:
        return jsonify({"status": "error", "message": "Error fetching cameras from the database"}), 500
    
    # 각 카메라별로 origin_video 데이터를 가져옴
    all_videos = []
    for camera in cameras:
        cam_num = camera['cam_num']
        cam_name = camera['cam_name']
        origin_videos = get_origin_videos_by_cam_num(cam_num)
        if origin_videos:
            all_videos.append({
                "cam_name": cam_name,
                "origin_videos": origin_videos
            })
    if all_videos is None:
        return jsonify({"status": "error", "message": "Error fetching data from the database"}), 500

    return jsonify({"message": "Image received and saved", "origin_videos": all_videos}), 200

# 7. 이전 분석 결과(영상) 불러오기 (Post)
@app.route('/realtime_search_previous_analysis_result_video', methods=['GET'])
def realtime_search_previous_analysis_result_video():
    or_video_id = request.args.get('or_video_id')

    # JSON 데이터가 제대로 수신되지 않았을 경우 확인
    if not or_video_id:
        return jsonify({"status": "error", "message": "No JSON data received"}), 400

    # or_video_id에 해당하는 pro_video_content 가져오기
    pro_video_content = get_pro_video_content(or_video_id)

    if pro_video_content is None:
        return jsonify({"status": "error", "message": "No processed video found for the given or_video_id"}), 404

    # 파일이 존재하는지 확인
    if not os.path.exists(pro_video_content):
        return jsonify({"status": "error", "message": "Video file not found"}), 404

    reencoded_path = f"{pro_video_content.rsplit('.', 1)[0]}_h264.mp4"
    print(f"Original video path: {pro_video_content}")  # 디버깅 메시지 추가
    print(f"Reencoded video path: {reencoded_path}")  # 디버깅 메시지 추가

    if not os.path.exists(reencoded_path):
        try:
            reencode_video(pro_video_content, reencoded_path)
            if not os.path.exists(reencoded_path):
                print(f"Reencoded file does not exist after re-encoding: {reencoded_path}")
                return jsonify({"error": "Failed to save re-encoded video"}), 500
        except subprocess.CalledProcessError as e:
            print(f"Error during re-encoding: {str(e)}")
            return jsonify({"error": "Error during re-encoding"}), 500

    # 파일을 스트리밍하여 클라이언트로 전송
    def generate():
        with open(reencoded_path, 'rb') as f:
            chunk = f.read(1024)
            while chunk:
                yield chunk
                chunk = f.read(1024)
    
    response = Response(generate(), content_type='video/mp4')
    response.headers["Content-Disposition"] = f"attachment; filename={os.path.basename(pro_video_content)}"
    return response

# 8. 이전 분석 결과(로그) 불러오기 (Post)
@app.route('/realtime_search_previous_analysis_result_logs', methods=['GET'])
def realtime_search_previous_analysis_result_logs():
    or_video_id = request.args.get('or_video_id')

    # JSON 데이터가 제대로 수신되지 않았을 경우 확인
    if not or_video_id:
        return jsonify({"status": "error", "message": "No JSON data received"}), 400

    log_datas = get_log_by_or_video_id(or_video_id)
    processed_data = process_log_data(log_datas)

    if log_datas is None:
        return jsonify({"status": "error", "message": "No processed video found for the given or_video_id"}), 404

    return jsonify({"message": "Analysis results successfully search logs", "logs" : processed_data}), 200

# 9. 이전 분석 결과 삭제 (Post)
@app.route('/realtime_delete_previous_analysis_result', methods=['DELETE'])
def realtime_delete_previous_analysis_result():
    data = request.get_json()

    # JSON 데이터가 제대로 수신되지 않았을 경우 확인
    if not data:
        return jsonify({"status": "error", "message": "No JSON data received"}), 400

    # 수신된 데이터가 문자열이 아닌 JSON 객체인지 확인
    if isinstance(data, str):
        return jsonify({"status": "error", "message": "Invalid JSON data format"}), 400

    user_data = data.get('user_data', {})
    or_video_id = user_data.get('or_video_id', '')

    # or_video_id에 해당하는 데이터 삭제
    if not delete_analysis_results(or_video_id):
        return jsonify({"status": "error", "message": "Error deleting data from the database"}), 500

    return jsonify({"message": "Analysis results successfully deleted"}), 200

if __name__ == '__main__':
    print("Starting Realtime Server")  # 서버 시작 디버깅 메시지
    app.run(host="0.0.0.0", port=5001, debug=True)