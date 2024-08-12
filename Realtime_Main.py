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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)

# folder_path에 있는 이미지 파일 수를 계산하는 함수
def count_images_in_folder(folder_path):
    # 이미지 파일 확장자 리스트
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    
    # 폴더 내 이미지 파일 수 계산
    image_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)])
    
    return image_count

# 텍스트 파일의 예측 결과를 요약하는 함수
def summarize_tracking_data(txt_file_path, cropped_images_dir):
    if not os.path.exists(txt_file_path):
        return "Making summary predictions...", "No image available"
    
    tracking_data = defaultdict(lambda: {
        'genders': [],
        'ages': [],
        'colors': [],
        'clothes': []
    })

    # txt 파일을 읽고 데이터를 추출합니다.
    with open(txt_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(', ')
            track_id = None
            gender = None
            age = None
            color = None
            clothes = None
            
            # 각 부분을 파싱합니다.
            for part in parts:
                if part.startswith("Track ID:"):
                    track_id = part.split(": ")[1]
                elif part.startswith("Gender:"):
                    gender = part.split(": ")[1]
                elif part.startswith("Age:"):
                    age = part.split(": ")[1]
                elif part.startswith("Color:"):
                    color = part.split(": ")[1]
                elif part.startswith("Clothes:"):
                    clothes = part.split(": ")[1]

            # Tracking ID별로 데이터를 수집합니다.
            if track_id is not None:
                tracking_data[track_id]['genders'].append(gender)
                tracking_data[track_id]['ages'].append(age)
                tracking_data[track_id]['colors'].append(color)
                tracking_data[track_id]['clothes'].append(clothes)

    # Tracking ID별로 가장 빈번한 값을 계산합니다.
    summary = []
    person_image_base64 = "No image available"  # 기본값 설정

    for track_id, data in tracking_data.items():
        most_common_gender = Counter(data['genders']).most_common(1)[0][0]
        most_common_age = Counter(data['ages']).most_common(1)[0][0]
        most_common_color = Counter(data['colors']).most_common(1)[0][0]
        most_common_clothes = Counter(data['clothes']).most_common(1)[0][0]

        summary.append(f"Track ID: {track_id}, Gender: {most_common_gender}, Age: {most_common_age}, "
                       f"Color: {most_common_color}, Clothes: {most_common_clothes}")
        
        # 해당 트랙 ID와 동일한 이름의 이미지가 있는지 확인
        image_filename = f"Person_{track_id}.jpg"
        image_path = os.path.join(cropped_images_dir, image_filename)
        if os.path.exists(image_path):
            person_image = cv2.imread(image_path)
            _, buffer = cv2.imencode('.jpg', person_image)
            person_image_base64 = base64.b64encode(buffer).decode('utf-8')
            break  # 첫 번째 매칭된 이미지를 저장한 후 루프 종료

    # 하나의 문자열로 합칩니다.
    return "\n".join(summary), person_image_base64

# 텍스트 파일의 마지막 줄을 읽어서 반환하는 함수
def get_last_line_from_txt(txt_file):
    if not os.path.exists(txt_file):
        return "Predicting..."

    with open(txt_file, 'rb') as f:
        f.seek(0, os.SEEK_END)  # 파일 끝으로 포인터 이동
        while f.tell() > 0:
            f.seek(-2, os.SEEK_CUR)  # 역순으로 읽기
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
        database='wb39_project',
        cursorclass=pymysql.cursors.DictCursor
    )

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

# 백그라운드에서 실행할 함수 정의
def run_background_process(user_id, filepath, processed_filepath):
    process = subprocess.Popen(
        ["python", "Realtime_Prediction.py", user_id, filepath, processed_filepath],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error occurred: {stderr.decode()}")

# 각 웹캠의 이미지가 저장될 폴더 경로 설정
SAVE_FOLDER = 'realtime_saved_images'
REALTIME_IMAGE_SAVE_PATH = 'realtime_uploaded_images'
WEBCAM_FOLDERS = [f"webcam_{i}" for i in range(4)]

# 실시간 분석 플래그
realtime_flag = False

# YOLO 모델 및 DeepSort 초기화
yolo_model = YOLO('models/yolov8x.pt')  # YOLO 모델 경로를 적절히 설정하세요
tracker = DeepSort(max_age=30, n_init=3, nn_budget=60)

# 폴더가 없으면 생성
for folder in WEBCAM_FOLDERS:
    os.makedirs(os.path.join(SAVE_FOLDER, folder), exist_ok=True)

# 0. 실시간 파일 업로드 엔드포인트 (Post)
@app.route('/realtime_upload_file', methods=['POST'])
def realtime_upload_file():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data received"}), 400
        if isinstance(data, str):
            return jsonify({"status": "error", "message": "Invalid JSON data format"}), 400

        user_data = data.get('user_data', {})
        user_id = user_data.get('user_id', '')

        filter_data = data.get('filter_data', {})
        top = filter_data.get('top', '')
        bottom = filter_data.get('bottom', '')

        user_image_path = os.path.join(REALTIME_IMAGE_SAVE_PATH, str(user_id))
        os.makedirs(user_image_path, exist_ok=True)

        connection = get_db_connection()
        filter_id = None

        with connection.cursor() as cursor:
            filter_sql = """
                INSERT INTO realtime_filter (filter_top, filter_bottom)
                VALUES (%s, %s)
            """
            cursor.execute(filter_sql, (top, bottom))
            filter_id = cursor.lastrowid
            print("filter DB create")
            print(f"filter ID : {filter_id}") #filter_id를 클라이언트에게 콜백으로 돌려줘야 함

            image_data = data.get('image_data', {})
            image_name = image_data.get('image_name', '')
            image_content_base64 = image_data.get('image_content', '')

            image_path = None
            if image_name and image_content_base64:
                image_content = base64.b64decode(image_content_base64)
                image_path = os.path.join(user_image_path, image_name).replace("\\", "/")

                with open(image_path, 'wb') as image_file:
                    image_file.write(image_content)
                print(f"Image: {image_name}")
                print(f"Image: {image_path}")

            connection.commit()
            connection.close()

            response = jsonify({"status": "success", "message": "Data received and processed successfully", "filter_id" : filter_id})
            response.status_code = 200
            return response

    except ValueError as e:
        print(f"A ValueError occurred: {str(e)}")
        return jsonify({"status": "error", "message": f"A ValueError occurred: {str(e)}"}), 400
    except KeyError as e:
        print(f"A KeyError occurred: {str(e)}")
        return jsonify({"status": "error", "message": f"A KeyError occurred: {str(e)}"}), 400
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500

# 0-1. 실시간 웹캠 이미지 전송 (Post)
@app.route('/upload_image_<int:webcam_id>', methods=['POST'])
def upload_image(webcam_id):
    # json_data를 가져와서 JSON 형식으로 파싱
    #json_data = request.form.get('json_data')
    
    #if not json_data:
        #return jsonify({"error": "No JSON data received"}), 400
    
    #try:
        # JSON 문자열을 Python 딕셔너리로 파싱
        #data = json.loads(json_data)
    #except json.JSONDecodeError:
        #return jsonify({"error": "Invalid JSON format"}), 400
    
    # user_id와 filter_id 추출
    #user_id = data.get('user_id')

    user_id = "admin"
    
    #if not user_id:
        #return jsonify({"error": "User ID is required"}), 400

    if webcam_id < 0 or webcam_id >= len(WEBCAM_FOLDERS):
        return jsonify({"error": "Invalid webcam ID"}), 400

    # 이미지 파일 수신
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Image decoding failed"}), 500

    # 이미지 파일 저장
    starttime = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{starttime}_{file.filename}.jpg"
    #folder_path = os.path.join(SAVE_FOLDER, WEBCAM_FOLDERS[webcam_id], user_id)
    folder_path = os.path.join(SAVE_FOLDER, WEBCAM_FOLDERS[webcam_id])
    processed_folder_path = os.path.join(SAVE_FOLDER, WEBCAM_FOLDERS[webcam_id], "processed_images")
    filepath = os.path.join(folder_path, filename).replace("\\", "/")
    processed_filepath = os.path.join(processed_folder_path, filename).replace("\\", "/")

    # 디렉토리가 존재하지 않을 경우 생성
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(processed_folder_path, exist_ok=True)

    cv2.imwrite(filepath, img)

    # 이미지 파일 수 계산
    image_count = count_images_in_folder(folder_path)

    print(f"Current count value: {image_count}")

    if image_count % 5 == 0 and image_count != 0:
        # 실시간 트래킹 처리
        person_detections = detect_persons(img, yolo_model)
        
        results = []
        for (xmin, ymin, xmax, ymax) in person_detections:
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], 1.0, 0])

        tracker_outputs = tracker.update_tracks(results, frame=img)

        # 트래킹 결과를 이미지에 시각화
        for track in tracker_outputs:
            bbox = track.to_tlbr()
            track_id = track.track_id
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(img, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 결과 이미지를 다시 저장
        cv2.imwrite(processed_filepath, img)

        # 결과 이미지를 Base64로 인코딩하여 클라이언트에 반환
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # 예측된 데이터 읽기
        output_txt_path = f"./realtime_saved_images/webcam_0/{user_id}/predictions.txt"
        predictions = get_last_line_from_txt(output_txt_path)

        print(f"Received and processed image from webcam {webcam_id} with shape: {img.shape} as {filename} / predictions : {predictions}")
        return jsonify({"message": "Image received and processed", "frame": image_base64, "data" : predictions}), 200

    if image_count % 72 == 0 and image_count != 0:
        print("Threading Start")

        # 새로운 스레드를 생성하여 백그라운드에서 스크립트를 실행
        threading.Thread(
            target=run_background_process,
            args=(user_id, filepath, processed_filepath)
        ).start()

        # 결과 이미지를 Base64로 인코딩하여 클라이언트에 반환
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        cropped_images_dir = f"./realtime_saved_images/webcam_0/{user_id}/cropped_images/"
        output_txt_path = f"./realtime_saved_images/webcam_0/{user_id}/predictions.txt"
        summary_predictions, person_image = summarize_tracking_data(output_txt_path, cropped_images_dir)

        print(f"Received and processed image from webcam {webcam_id} with shape: {img.shape} as {filename} / Summary Predictions : {summary_predictions}")
        return jsonify({"message": "Image received and processed", "frame": image_base64, "sum_data": summary_predictions, "person_image" : person_image}), 200

    # 결과 이미지를 Base64로 인코딩하여 클라이언트에 반환
    _, buffer = cv2.imencode('.jpg', img)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # 예측된 데이터 읽기
    output_txt_path = f"./realtime_saved_images/webcam_0/{user_id}/predictions.txt"
    predictions = get_last_line_from_txt(output_txt_path)

    print(f"Received and return image from webcam {webcam_id} with shape: {img.shape} as {filename} / predictions : {predictions}")
    return jsonify({"message": "Image received and processed", "frame": image_base64, "data" : predictions}), 200

# 0-2. 실시간 웹캠 이미지 전송 종료 (Post)
@app.route('/realtime_upload_image_end_<int:webcam_id>', methods=['POST'])
def realtime_upload_image_end(webcam_id):
    data = request.get_json()

    # JSON 데이터가 제대로 수신되지 않았을 경우 확인
    if not data:
        return jsonify({"status": "error", "message": "No JSON data received"}), 400

    # 수신된 데이터가 문자열이 아닌 JSON 객체인지 확인
    if isinstance(data, str):
        return jsonify({"status": "error", "message": "Invalid JSON data format"}), 400

    user_data = data.get('user_data', {})
    user_id = user_data.get('user_id', '')

    if webcam_id < 0 or webcam_id >= len(WEBCAM_FOLDERS):
        return jsonify({"error": "Invalid webcam ID"}), 400

    # Processed 이미지 폴더 경로 설정
    processed_folder_path = os.path.join(SAVE_FOLDER, WEBCAM_FOLDERS[webcam_id], "processed_images")
    origin_folder_path = os.path.join(SAVE_FOLDER, WEBCAM_FOLDERS[webcam_id])

    # 폴더가 존재하는지 확인
    if not os.path.exists(processed_folder_path):
        return jsonify({"status": "error", "message": "Processed folder not found"}), 404

    # 이미지 파일 목록 가져오기
    pro_image_files = sorted([f for f in os.listdir(processed_folder_path) if f.endswith('.jpg')])
    or_image_files = sorted([f for f in os.listdir(origin_folder_path) if f.endswith('.jpg')])

    # 이미지가 있는지 확인
    if len(pro_image_files) == 0:
        return jsonify({"status": "error", "message": "No images found to create video"}), 404

    # 첫 번째 이미지를 읽어 크기를 가져옴
    first_image_path = os.path.join(processed_folder_path, pro_image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # 비디오 작성 준비
    video_path = os.path.join(processed_folder_path, f"{user_id}_output_video.mp4")
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    # 이미지들을 비디오에 작성
    for image_file in pro_image_files:
        image_path = os.path.join(processed_folder_path, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    video_writer.release()

    # 이미지 삭제(처리된 이미지)
    for image_file in pro_image_files:
        image_path = os.path.join(processed_folder_path, image_file)
        os.remove(image_path)

    # 이미지 삭제(원본 이미지)
    for image_file in or_image_files:
        image_path = os.path.join(origin_folder_path, image_file)
        os.remove(image_path)
    
    print(f"Received and saved image from webcam {webcam_id} End")

    return jsonify({"message": "Image received and saved"}), 200


if __name__ == '__main__':
    print("Starting Realtime Server")  # 서버 시작 디버깅 메시지
    app.run(host="0.0.0.0", port=5001, debug=True)