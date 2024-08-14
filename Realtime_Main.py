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

# 각 웹캠의 이미지가 저장될 폴더 경로 설정
SAVE_FOLDER = 'realtime_saved_images'

# 처리 시작 시간
starttime = ""

#예측 중인지 확인하는 변수
process_playing = False

# YOLO 모델 및 DeepSort 초기화
yolo_model = YOLO('models/yolov8x.pt')  # YOLO 모델 경로를 적절히 설정하세요
tracker = DeepSort(max_age=30, n_init=3, nn_budget=60)

# folder_path에 있는 이미지 파일 수를 계산하는 함수
def count_images_in_folder(folder_path):
    # 이미지 파일 확장자 리스트
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    
    # 폴더 내 이미지 파일 수 계산
    image_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)])
    
    return image_count

# user_id로 user_no 정보 가져오기
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

# user_id로 user_no 정보 가져오기
def get_cam_num(user_no, cam_name):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT cam_num FROM camera WHERE user_no = %s AND cam_name = %s"
            cursor.execute(sql, (user_no, cam_name))
            result = cursor.fetchone()
            if result:
                return result['cam_num']
            else:
                print(f"No record found for user_id: {user_no}")
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

# 영상 및 log 결과 DB에 저장
def save_video_to_db(user_no, cam_num, start_time, origin_video_path, processed_video_path, output_txt_path, processed_log_txt_path, cropped_images_path):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # 원본 비디오를 origin_video 테이블에 저장
            origin_video_name = os.path.basename(origin_video_path)
            insert_origin_video_query = """
                INSERT INTO origin_video (or_video_name, or_video_content, start_time, cam_num)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_origin_video_query, (origin_video_name, origin_video_path, start_time, cam_num))
            or_video_id = cursor.lastrowid  # 방금 삽입된 원본 비디오의 ID

            # 처리된 비디오를 processed_video 테이블에 저장
            processed_video_name = os.path.basename(processed_video_path)
            insert_processed_video_query = """
                INSERT INTO processed_video (or_video_id, pro_video_name, pro_video_content, user_no)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_processed_video_query, (or_video_id, processed_video_name, processed_video_path, user_no))

            # txt 파일 내용을 log 테이블에 저장
            with open(output_txt_path, 'r') as txt_file:
                for line in txt_file:
                    log_data = line.strip()
                    insert_log_query = """
                        INSERT INTO log (log_data, user_no, cam_num, or_video_id)
                        VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(insert_log_query, (log_data, user_no, cam_num, or_video_id))

            # 변경 사항 커밋
            connection.commit()

            print("Videos and logs have been saved to the database.")

            os.remove(output_txt_path)
            os.remove(processed_log_txt_path)
            os.remove(cropped_images_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        connection.rollback()
    finally:
        connection.close()

# 텍스트 파일의 예측 결과를 요약하는 함수
def summarize_tracking_data(txt_file_path, cropped_images_dir):
    if not os.path.exists(txt_file_path):
        return "Making summary predictions...", "No image available"
    
    tracking_data = defaultdict(lambda: {
        'gender': [],
        'age': [],
        'upclothes': [],
        'downclothes': [],
    })

    # txt 파일을 읽고 데이터를 추출합니다.
    with open(txt_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(', ')
            track_id = None
            gender = None
            age = None
            upclothes = None
            downclothes = None
            
            # 각 부분을 파싱합니다.
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

            # Tracking ID별로 데이터를 수집합니다.
            if track_id is not None:
                tracking_data[track_id]['gender'].append(gender)
                tracking_data[track_id]['age'].append(age)
                tracking_data[track_id]['upclothes'].append(upclothes)
                tracking_data[track_id]['downclothes'].append(downclothes)

    # Tracking ID별로 가장 빈번한 값을 계산합니다.
    summary = []
    person_image_base64 = "No image available"  # 기본값 설정

    def get_most_common_value(values):
        # 빈도수를 계산하여 가장 빈번한 값을 반환
        counter = Counter(values)
        most_common = counter.most_common()
        
        # 가장 빈번한 값이 "unknown"이면, 그 다음 빈번한 값을 반환
        for value, _ in most_common:
            if value != "unknown":
                return value
        return "unknown"

    for track_id, data in tracking_data.items():
        most_common_gender = get_most_common_value(data['gender'])
        most_common_age = get_most_common_value(data['age'])
        most_common_upclothes = get_most_common_value(data['upclothes'])
        most_common_downclothes = get_most_common_value(data['downclothes'])

        summary.append(f"Track ID: {track_id}, Gender: {most_common_gender}, Age: {most_common_age}, "
                       f"Upclothes: {most_common_upclothes}, Downclothes: {most_common_downclothes}")
        
        # 해당 트랙 ID와 동일한 이름의 이미지가 있는지 확인
        image_filename = f"Person_{track_id}.jpg"
        image_path = os.path.join(cropped_images_dir, image_filename)
        if os.path.exists(image_path):
            person_image = cv2.imread(image_path)
            _, buffer = cv2.imencode('.jpg', person_image)
            person_image_base64 = base64.b64encode(buffer).decode('utf-8')

    # 하나의 문자열로 합칩니다.
    return "\n".join(summary), person_image_base64

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
def run_background_process(user_id, user_cam_folder_path, origin_filepath):
    global process_playing
    process = subprocess.Popen(
        ["python", "Realtime_Prediction.py", user_id, user_cam_folder_path, origin_filepath],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    process_playing = False

    if process.returncode != 0:
        print(f"Error occurred: {stderr.decode()}")

# 0. 실시간 분석 정보 업로드 엔드포인트 (Post)
@app.route('/realtime_upload_file', methods=['POST'])
def realtime_upload_file():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data received"}), 400
        if isinstance(data, str):
            return jsonify({"status": "error", "message": "Invalid JSON data format"}), 400

        user_data = data.get('user_data', {})
        #user_id = user_data.get('user_id', '')
        #cam_name = user_data.get('cam_name', '')

        user_id = "admin"
        cam_name = "우송대"

        connection = get_db_connection()

        with connection.cursor() as cursor:
            # 1. 사용자 추가 또는 조회 (user_id는 고유해야 함)
            cursor.execute("SELECT user_no FROM user WHERE user_id = %s", (user_id,))
            user_row = cursor.fetchone()
            if user_row is None:
                cursor.execute("INSERT INTO user (user_id) VALUES (%s)", (user_id,))
                user_no = cursor.lastrowid
            else:
                user_no = user_row['user_no']

            # 2. 해당 user_id에 대한 cam_name 추가 또는 조회
            cursor.execute("SELECT cam_num FROM camera WHERE cam_name = %s AND user_no = %s", (cam_name, user_no))
            cam_row = cursor.fetchone()
            if cam_row is None:
                cursor.execute("INSERT INTO camera (cam_name, user_no) VALUES (%s, %s)", (cam_name, user_no))
                cam_num = cursor.lastrowid
            else:
                cam_num = cam_row['cam_num']

        connection.commit()
        connection.close()

        return jsonify({"status": "success", "message": "User and Camera added successfully", "user_no": user_no, "cam_num": cam_num}), 200

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
@app.route('/realtime_upload_image', methods=['POST'])
def upload_image():
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
    #cam_name = data.get('cam_name')
    user_id = "admin"
    cam_name = "우송대"
    #if not user_id:
        #return jsonify({"error": "User ID is required"}), 400

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
    cropped_images_path = os.path.join(user_cam_folder_path, "cropped_images")
    output_txt_path = os.path.join(user_cam_folder_path, "predictions.txt")

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
            args=(user_id, user_cam_folder_path, origin_filepath)
        ).start()
        process_playing = True


    if image_count % 5 == 0 and image_count % 150 != 0:
        print("Process Start")
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
        predictions = get_last_line_from_txt(output_txt_path)

        print(f"Received and processed image from cam {cam_name} with shape: {img.shape} as {filename} / predictions : {predictions}")
        return jsonify({"message": "Image received and processed", "frame": image_base64, "person_data" : predictions}), 200

    if image_count % 150 == 0 and image_count != 0:
        # 결과 이미지를 Base64로 인코딩하여 클라이언트에 반환
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        summary_predictions, person_image = summarize_tracking_data(output_txt_path, cropped_images_path)

        print(f"Received and processed image from cam {cam_name} with shape: {img.shape} as {filename} / Summary Predictions : {summary_predictions}")
        return jsonify({"message": "Image received and processed", "frame": image_base64, "person_data": summary_predictions, "person_image" : person_image}), 200

    # 결과 이미지를 Base64로 인코딩하여 클라이언트에 반환
    _, buffer = cv2.imencode('.jpg', img)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # 예측된 데이터 읽기
    predictions = get_last_line_from_txt(output_txt_path)

    print(f"Received and return image from cam {cam_name} with shape: {img.shape} as {filename} / predictions : {predictions}")
    return jsonify({"message": "Image received and processed", "person_data" : predictions}), 200

# 0-2. 실시간 웹캠 이미지 전송 종료 (Post)
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
    #cam_name = user_data.get('cam_name', '')
    cam_name = "우송대"

    user_no = get_user_no(user_id)
    cam_num = get_cam_num(user_id, cam_name)
    start_time = convert_filename_to_datetime(or_image_files[0])

    # Processed 이미지 폴더 경로 설정
    user_folder_path = os.path.join(SAVE_FOLDER, user_id)
    user_cam_folder_path = os.path.join(user_folder_path, cam_name)
    save_origin_path = os.path.join(user_cam_folder_path, "origin_video")
    save_processed_path = os.path.join(user_cam_folder_path, "processed_video")
    origin_folder_path = os.path.join(user_cam_folder_path, "origin_images")
    processed_folder_path = os.path.join(user_cam_folder_path, "processed_images")
    cropped_images_path = os.path.join(user_cam_folder_path, "cropped_images")
    output_txt_path = os.path.join(user_cam_folder_path, "predictions.txt")
    processed_log_txt_path = os.path.join(user_folder_path, "processed_images.log")
    
    os.makedirs(save_origin_path, exist_ok=True)
    os.makedirs(save_processed_path, exist_ok=True)

    # 이미지 파일 목록 가져오기
    or_image_files = sorted([f for f in os.listdir(origin_folder_path) if f.endswith('.jpg')])
    pro_image_files = sorted([f for f in os.listdir(processed_folder_path) if f.endswith('.jpg')])

    # 이미지가 있는지 확인
    if len(or_image_files) == 0:
        return jsonify({"status": "error", "message": "No images found to create video"}), 404

    if len(pro_image_files) == 0:
        return jsonify({"status": "error", "message": "No images found to create video"}), 404

    # 첫 번째 이미지를 읽어 크기를 가져옴
    first_image_path = os.path.join(origin_folder_path, or_image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # 원본 비디오 작성 준비
    origin_video_path = os.path.join(save_origin_path, f"{cam_name}_{start_time}_original_video.mp4")
    origin_video_writer = cv2.VideoWriter(origin_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    # 원본 이미지를 사용하여 원본 비디오 생성
    for image_file in or_image_files:
        image_path = os.path.join(origin_folder_path, image_file)
        image = cv2.imread(image_path)
        origin_video_writer.write(image)

    origin_video_writer.release()

    # 원본 비디오 작성 완료
    print(f"Original video created at {origin_video_writer}")

    # 합쳐진 비디오 작성 준비
    processed_video_path = os.path.join(save_processed_path, f"{cam_name}_{start_time}_processed_video.mp4")
    processed_video_writer = cv2.VideoWriter(processed_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    # 이미지들을 합쳐진 비디오에 작성 (원본 이미지와 처리된 이미지를 비교)
    for image_file in pro_image_files:
        image_path = os.path.join(processed_folder_path, image_file)
        image = cv2.imread(image_path)
        processed_video_writer.write(image)

    processed_video_writer.release()

    # 합쳐진 비디오 작성 완료
    print(f"Combined video created at {processed_video_path}")

    # 이미지 삭제(처리된 이미지)
    for image_file in pro_image_files:
        image_path = os.path.join(processed_folder_path, image_file)
        os.remove(image_path)

    # 이미지 삭제(원본 이미지)
    for image_file in or_image_files:
        image_path = os.path.join(origin_folder_path, image_file)
        os.remove(image_path)
    
    save_video_to_db(user_no, cam_num, start_time, origin_video_path, processed_video_path, output_txt_path, processed_log_txt_path, cropped_images_path)
    
    print(f"Received and saved image from cam {cam_name} End")

    return jsonify({"message": "Image received and saved"}), 200

if __name__ == '__main__':
    print("Starting Realtime Server")  # 서버 시작 디버깅 메시지
    app.run(host="0.0.0.0", port=5001, debug=True)