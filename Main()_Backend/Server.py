from flask import Flask, request, jsonify, Response, url_for
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os, base64
import pymysql
import subprocess
import threading
import re
import numpy as np
import cv2
from datetime import datetime
from filelock import FileLock
import concurrent.futures
import math
from moviepy.editor import VideoFileClip
import json
import time
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)

# 데이터베이스 연결 함수
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='1234',
        database='server',
        cursorclass=pymysql.cursors.DictCursor
    )

# 파일 저장 경로 설정
VIDEO_SAVE_PATH = 'uploaded_videos'
IMAGE_SAVE_PATH = 'uploaded_images'

# 디렉토리 생성
os.makedirs(VIDEO_SAVE_PATH, exist_ok=True)
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

socketio = SocketIO(app)

# 작업 상태를 추적하는 전역 변수 추가
task_status = {}
task_lock = threading.Lock()

# 현재 작업 상태 키를 가져옴
def get_task_key(user_id, video_name, filter_id):
    return f"{user_id}_{video_name}_{filter_id}"

# 현재 작업 상태 키를 설정함
def set_task_status(task_key, status):
    with task_lock:
        task_status[task_key] = {'status': status}

# 현재 작업 상태를 가져옴
def get_task_status(task_key):
    with task_lock:
        return task_status.get(task_key, {}).get('status')

# 트래킹 작업 상태 관리
class TrackingTaskManager:
    def __init__(self, missing_videos, callback):
        self.missing_videos = missing_videos
        self.callback = callback
        self.completed_tasks = 0
        self.total_tasks = len(missing_videos)
        self.lock = threading.Lock()
        self.callback_called = False  # 콜백 함수가 호출되었는지 추적
        print(f"Total tasks set to {self.total_tasks}")  # 디버깅 메시지 추가

    def task_completed(self):
        with self.lock:
            self.completed_tasks += 1
            print(f"Task completed: {self.completed_tasks}/{self.total_tasks}")  # 디버깅 메시지 추가
            if self.completed_tasks == self.total_tasks and not self.callback_called:
                if callable(self.callback):
                    print("Executing callback")  # 디버깅 메시지 추가
                    self.callback()
                    self.callback_called = True  # 콜백 함수가 다시 호출되지 않도록 설정


class TrackingTaskManager2:
    def __init__(self, video_names_for_clip_process, callback):
        self.video_names_for_clip_process = video_names_for_clip_process
        self.callback = callback
        self.completed_tasks = 0
        self.total_tasks = len(video_names_for_clip_process)
        self.lock = threading.Lock()
        self.callback_called = False
        print(f"Total tasks set to {self.total_tasks}")

    def task_completed(self):
        with self.lock:
            self.completed_tasks += 1
            print(f"Task completed: {self.completed_tasks}/{self.total_tasks}")
            if self.completed_tasks >= self.total_tasks and not self.callback_called:
                if callable(self.callback):
                    print("Executing callback")
                    self.callback()
                    self.callback_called = True


def task_completed(self):
    self.completed_tasks += 1
    if self.completed_tasks >= self.total_tasks:
        print(f"All tasks completed: {self.completed_tasks}/{self.total_tasks}. Proceeding to clip creation.")
        # After all tracking tasks are completed, create the clip videos
        self.final_task()  # This will call clip_video with the necessary parameters
    else:
        print(f"Task completed: {self.completed_tasks}/{self.total_tasks}")

#비디오 타임스탬프 가져옴
def get_video_metadata(video_path):
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'format_tags=creation_time', '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode('utf-8').strip()
        return output if output else None
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving video metadata: {e.output.decode('utf-8')}")
        return None

#타임 스탬프를 Datetime 형식으로 변환
def convert_timestamp_format(timestamp_str):
    # ISO 8601 형식의 문자열을 파싱하여 datetime 객체로 변환
    dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    
    # 시간대를 고려하여 필요한 만큼의 시간차를 더함 (여기서는 예제로 9시간을 더함)
    dt = dt + timedelta(hours=9)  # UTC+9 (KST 시간대로 변환)

    # 원하는 형식으로 datetime 객체를 문자열로 변환
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# person_id를 기반으로 person_no 리스트를 찾기
def get_person_nos(person_id, user_no, filter_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            person_sql = "SELECT person_no FROM person WHERE person_id = %s AND user_no = %s AND filter_id = %s"
            cursor.execute(person_sql, (person_id, user_no, filter_id))
            person_results = cursor.fetchall()
            
            if person_results:
                return [row['person_no'] for row in person_results]
            else:
                return []
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return []
    finally:
        connection.close()

# person_id를 기반으로 person_no 리스트를 찾기
def get_person_nos2(person_id, user_no):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            person_sql = "SELECT person_no FROM person WHERE person_id = %s AND user_no = %s"
            cursor.execute(person_sql, (person_id, user_no))
            person_results = cursor.fetchall()
            
            if person_results:
                return [row['person_no'] for row in person_results]
            else:
                return []
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return []
    finally:
        connection.close()

# user_no , video_name 기반 filter_id 조회
def get_vidoe_names_by_or_video_id_and_user_no_and_filter_id(or_video_id, user_no, filter_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT pro_video_name
                FROM processed_video
                WHERE or_video_id = %s AND user_no = %s AND filter_id = %s
            """
            cursor.execute(sql, (or_video_id, user_no, filter_id))
            result = cursor.fetchall()
            if result:
                return [os.path.splitext(row['pro_video_name'])[0].replace('_output', '') for row in result]
            else:
                return []
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return []
    finally:
        connection.close()

# user_no , person_id 기반 filter_id 조회
def get_or_video_id_by_person_id_and_user_no_and_filter_id(person_id, user_no, filter_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT or_video_id
                FROM person
                WHERE person_id = %s AND user_no = %s AND filter_id = %s
            """
            cursor.execute(sql, (person_id, user_no, filter_id))
            result = cursor.fetchall()
            if result:
                return [row['or_video_id'] for row in result]
            else:
                return []
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return []
    finally:
        connection.close()

# 위도와 경도로 거리 계산
def haversine(lat1, lon1, lat2, lon2):
    # 지구의 반지름 (킬로미터 단위)
    R = 6371.0
    
    # 위도와 경도를 라디안 단위로 변환
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # 하버사인 공식 적용
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # 거리 계산 (미터 단위로 변환)
    distance = R * c * 1000
    
    return distance

# 계산된 거리의 총 합
def total_distance(coordinates, path):
    total_dist = 0.0
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        lat1, lon1 = coordinates[start]
        lat2, lon2 = coordinates[end]
        total_dist += haversine(lat1, lon1, lat2, lon2)
    return total_dist

# 시간 구하기
def calculate_radius(start_time, end_time, total_distance, current_time):
    # 시간 문자열을 datetime 객체로 변환
    start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    current = current_time

    # 관찰 기간 동안의 평균 속도 계산 (m/s)
    observation_duration = (end - start).total_seconds()
    average_speed = total_distance / observation_duration

    # 마지막 관찰 이후 경과 시간 계산
    time_since_last_observation = (current - end).total_seconds()

    # 최대 이동 거리 계산 (원의 반지름)
    radius = average_speed * time_since_last_observation

    return radius

# 동영상의 길이 구하기
def get_video_duration(video_path):
    with VideoFileClip(video_path) as video:
        return video.duration

# 비디오 파일의 길이를 계산하는 함수
def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = frame_count / fps
    cap.release()
    return length

# user_no , video_name 기반 filter_id 조회
def get_filter_ids_by_video_name_and_user(video_name, user_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # video_name의 확장자를 제거하고 _output.mp4를 추가
            base_video_name = os.path.splitext(video_name)[0]
            output_video_name = f"{base_video_name}_output.mp4"
            
            sql = """
                SELECT filter_id
                FROM processed_video
                WHERE pro_video_name = %s AND user_no = (SELECT user_no FROM user WHERE user_id = %s)
            """
            cursor.execute(sql, (output_video_name, user_id))
            result = cursor.fetchall()
            if result:
                return [row['filter_id'] for row in result]
            else:
                return []
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return []
    finally:
        connection.close()
    
#pro_video_ids에 해당하는 pro_video_names 가져오기
def get_pro_video_names_by_ids(pro_video_ids):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # pro_video_ids가 빈 리스트일 경우 처리
            if not pro_video_ids:
                return []

            format_strings = ','.join(['%s'] * len(pro_video_ids))
            sql = f"""
                SELECT pro_video_name
                FROM processed_video
                WHERE pro_video_id IN ({format_strings})
            """
            cursor.execute(sql, pro_video_ids)
            result = cursor.fetchall()
            
            # 파일 이름만 추출하여 리스트에 담기
            pro_video_names = [row['pro_video_name'].rsplit('_output', 1)[0] for row in result]
            return pro_video_names

    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return []

    finally:
        connection.close()

# filter_id에 해당하는 pro_video_id 가져오기
def get_pro_video_ids_by_filter_id(filter_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT pro_video_id
                FROM processed_video
                WHERE filter_id = %s
            """
            cursor.execute(sql, (filter_id,))
            result = cursor.fetchall()
            pro_video_ids = [row['pro_video_id'] for row in result]
            return pro_video_ids

    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return []

    finally:
        connection.close()

# video_name과 user_no, or_video_id 에 해당하는 filter_id 가져오기
def get_filter_id_by_video_name_and_user_no_and_or_video_id(or_video_id, video_name, user_no):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            pro_video_name = f"{video_name}_output.mp4"
            sql = """
                SELECT filter_id
                FROM processed_video
                WHERE or_video_id = %s AND pro_video_name = %s AND user_no = %s
            """
            cursor.execute(sql, (or_video_id, pro_video_name, user_no))
            result = cursor.fetchone()
            if result:
                return result['filter_id']
            else:
                return None
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

# 여러 pro_video_id에 대한 or_video_id를 가져오는 함수
def get_or_video_ids_by_pro_video_ids(pro_video_ids):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            format_strings = ','.join(['%s'] * len(pro_video_ids))
            sql = f"""
                SELECT pro_video_id, or_video_id
                FROM processed_video
                WHERE pro_video_id IN ({format_strings})
            """
            cursor.execute(sql, tuple(pro_video_ids))
            results = cursor.fetchall()
            return {row['pro_video_id']: row['or_video_id'] for row in results}
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return {}
    finally:
        connection.close()

# person_no 에 해당하는 필터 정보 가져오기
def get_filter_id_by_person_no(person_no):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT filter_id FROM person WHERE person_no = %s"
            cursor.execute(sql, (person_no,))
            result = cursor.fetchone()
            if result:
                return result['filter_id']
            else:
                print(f"No record found for person_no: {person_no}")
                return None
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

# 클립추출을 위한 트래킹 영상이 존재하는지 확인
def does_video_file_exist(user_id, video_name, person_id, filter_id):
    video_dir = f'./extracted_images/{user_id}/filter_{filter_id}/{video_name}_clip/person_{person_id}/'
    if not os.path.exists(video_dir):
        print(f"Directory does not exist: {video_dir}")
        return False

    # 디렉토리 내의 비디오 파일 찾기 (.mp4 확장자)
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4') and f'{video_name}_person_{person_id}' in f]
    if not video_files:
        print(f"No video files found in directory: {video_dir}")
        return False

    return True

#person_no로 or_video_id 가져오는 메서드
def get_or_video_id_by_person_no(person_no):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT or_video_id FROM person WHERE person_no = %s"
            cursor.execute(sql, (person_no,))
            result = cursor.fetchone()
            if result:
                return result['or_video_id']
            else:
                print(f"No record found for person_no: {person_no}")
                return None
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

#Stream_video를 위한 비디오 파일들 병합
def merge_videos(video_paths, output_path):
    # 비디오 개수에 따른 그리드 설정
    num_videos = len(video_paths)
    
    if num_videos == 1:
        # 비디오가 1개일 경우, 그대로 출력
        command = [
            'ffmpeg', '-i', video_paths[0],
            '-c:v', 'libx264', '-crf', '23', '-preset', 'veryfast', output_path
        ]
    elif num_videos == 2:
        # 비디오가 2개일 경우, hstack으로 수평 병합
        command = [
            'ffmpeg',
            '-i', video_paths[0],
            '-i', video_paths[1],
            '-filter_complex', '[0:v][1:v]hstack=inputs=2',
            '-c:v', 'libx264', '-crf', '23', '-preset', 'veryfast', output_path
        ]
    elif num_videos == 3:
        # 비디오가 3개일 경우, 첫 번째 비디오는 2x1의 첫 번째 줄, 나머지 2개는 2x1의 두 번째 줄로 구성
        command = [
            'ffmpeg',
            '-i', video_paths[0],
            '-i', video_paths[1],
            '-i', video_paths[2],
            '-filter_complex',
            '[1:v][2:v]hstack=inputs=2[bottom];[0:v][bottom]vstack=inputs=2',
            '-c:v', 'libx264', '-crf', '23', '-preset', 'veryfast', output_path
        ]
    elif num_videos == 4:
        # 비디오가 4개일 경우, 2x2 그리드로 병합
        command = [
            'ffmpeg',
            '-i', video_paths[0],
            '-i', video_paths[1],
            '-i', video_paths[2],
            '-i', video_paths[3],
            '-filter_complex',
            '[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2',
            '-c:v', 'libx264', '-crf', '23', '-preset', 'veryfast', output_path
        ]
    else:
        raise ValueError("This function supports between 1 and 4 videos.")

    subprocess.run(command, check=True)

# pro_video_content를 DB에서 얻어오는 함수
def get_video_path(user_id, filter_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # 사용자 번호 가져오기
            cursor.execute("SELECT user_no FROM user WHERE user_id = %s", (user_id,))
            user = cursor.fetchone()
            if not user:
                return None
            user_no = user['user_no']
            
            # 특정 필터 ID에 해당하는 모든 동영상 경로 가져오기
            cursor.execute("SELECT pro_video_content FROM processed_video WHERE user_no = %s AND filter_id = %s", (user_no, filter_id))
            videos = cursor.fetchall()
            if not videos:
                return None
            
            # 동영상 경로들을 리스트로 반환
            return [video['pro_video_content'] for video in videos]
    finally:
        connection.close()

#person 정보별 이미지 가져오기
def get_image_paths_for_person_nos(person_nos):
    connection = get_db_connection()
    image_paths = []
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT person_no, person_origin_face
                FROM person
                WHERE person_no IN (%s)
            """ % ','.join(['%s'] * len(person_nos))
            cursor.execute(sql, person_nos)
            results = cursor.fetchall()
            image_paths = {row['person_no']: row['person_origin_face'] for row in results}
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
    finally:
        connection.close()
    return image_paths

# 트리거처럼 동작하도록 처리 (새로운 클립 추가 시 호출)
def update_person_face_from_clip(person_no):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # person_no로 person_id 가져오기
            sql = "SELECT person_id, or_video_id FROM person WHERE person_no = %s"
            cursor.execute(sql, (person_no,))
            person_result = cursor.fetchone()
            if not person_result:
                print(f"No person_id found for person_no: {person_no}")
                return

            person_id = person_result['person_id']
            or_video_id = person_result['or_video_id']
            
            # 원본 비디오 이름을 가져옴
            sql = """
                SELECT or_video_name 
                FROM origin_video 
                WHERE or_video_id = %s
            """
            cursor.execute(sql, (or_video_id,))
            or_video_name_result = cursor.fetchone()
            if not or_video_name_result:
                print(f"No or_video_name found for or_video_id: {or_video_id}")
                return

            # .mp4 확장자 제거
            or_video_name = os.path.splitext(or_video_name_result['or_video_name'])[0]
            
            # user_id를 가져옴
            sql = "SELECT user_id FROM user WHERE user_no = (SELECT user_no FROM person WHERE person_no = %s)"
            cursor.execute(sql, (person_no,))
            user_result = cursor.fetchone()
            if not user_result:
                print(f"No user found for person_no: {person_no}")
                return

            user_id = user_result['user_id']

            

            # person_origin_face 설정
            user_image_dir = f'./uploaded_images/{user_id}/'
            if not os.path.exists(user_image_dir):
                print(f"Directory not found: {user_image_dir}")
                return

            # 업로드된 이미지 파일 찾기
            user_image_files = [f for f in os.listdir(user_image_dir) if f.endswith('.jpg') or f.endswith('.png')]
            if not user_image_files:
                print(f"No uploaded image files found in directory: {user_image_dir}")
                return

            # 첫 번째 업로드된 이미지 파일 사용 (필요에 따라 선택 방법 변경 가능)
            uploaded_image_name = user_image_files[0]
            person_face_relative_path = os.path.join(user_image_dir, uploaded_image_name).replace("\\", "/")

            # person 테이블 업데이트
            sql = """
                UPDATE person 
                SET person_face = %s 
                WHERE person_no = %s
            """
            cursor.execute(sql, (person_face_relative_path, person_no))
            connection.commit()
            print(f"Updated person_face for person_no: {person_no}")

    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
    finally:
        connection.close()

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

# or_video_id 통해 pro_video_id 정보 가져오기
def get_pro_video_id(or_video_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT pro_video_id FROM processed_video WHERE or_video_id = %s"
            cursor.execute(sql, (or_video_id,))
            result = cursor.fetchone()
            if result:
                return result['pro_video_id']
            else:
                print(f"No record found for or_video_id: {or_video_id}")
                return None
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

# 절대 경로로 or_video_id 가져오기
def get_or_video_id_by_path(video_path):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT or_video_id FROM origin_video WHERE or_video_content = %s"
            cursor.execute(sql, (video_path,))
            result = cursor.fetchone()
            if result:
                return result['or_video_id']
            else:
                print(f"No record found for video_path: {video_path}")
                return None
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

# or_video_id로 or_video_name 가져오기
def get_or_video_name(or_video_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # or_video_id를 통해 or_video_name 조회
            sql = """
                SELECT or_video_name 
                FROM origin_video 
                WHERE or_video_id = %s
            """
            cursor.execute(sql, (or_video_id,))
            result = cursor.fetchone()
            if result:
                # .mp4 확장자 제거
                return os.path.splitext(result['or_video_name'])[0]
            else:
                print(f"No or_video_name found for or_video_id: {or_video_id}")
                return None
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

# 데이터베이스에서 filter 정보 가져오기
def get_filter_info(filter_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM filter WHERE filter_id = %s"
            cursor.execute(sql, (filter_id,))
            result = cursor.fetchone()
            return result if result else None
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

# 메모장에서 데이터 읽기 및 파싱 함수
def parse_info_file(file_path):
    person_info = []
    with open(file_path, 'r') as file:
        content = file.read()
        persons = content.split('person_')[1:]
        for person in persons:
            person_id_match = re.search(r'(\d+):', person)
            gender_match = re.search(r'gender: (\w+)', person)
            age_match = re.search(r'age: (\w+)', person)
            upclothes_match = re.search(r'upclothes: (\w+)', person)
            downclothes_match = re.search(r'downclothes: (\w+)', person)
            if person_id_match and gender_match and age_match:
                person_id = person_id_match.group(1)
                gender = gender_match.group(1)
                age = age_match.group(1)
                upclothes = upclothes_match.group(1)
                downclothes = downclothes_match.group(1)
                person_info.append({
                    'person_id': person_id,
                    'gender': gender,
                    'age': age,
                    'upclothes': upclothes,
                    'downclothes': downclothes
                })
    return person_info

# Person List DB 저장 (이미지 없을 때)
def save_to_db(person_info, or_video_id, user_id, user_no, filter_id):
    connection = get_db_connection()
    saved_paths = []
    try:
        with connection.cursor() as cursor:
            for person in person_info:
                or_video_name_result = get_or_video_name(or_video_id)
                if not or_video_name_result:
                    print(f"No or_video_name found for or_video_id: {or_video_id}")
                    continue

                or_video_name = os.path.splitext(or_video_name_result)[0]

                person_id = person['person_id']
                person_age = person['age']
                person_gender = person['gender']
                person_upclothes = person['upclothes']
                person_downclothes = person['downclothes']

                # 중복 체크
                check_sql = """
                    SELECT person_origin_face FROM person
                    WHERE person_age = %s
                    AND person_gender = %s
                    AND person_upclothes = %s
                    AND person_downclothes = %s
                    AND user_no = %s
                    AND filter_id = %s
                    AND or_video_id = %s
                """
                cursor.execute(check_sql, (
                    person_age,
                    person_gender,
                    person_upclothes,
                    person_downclothes,
                    user_no,
                    filter_id,
                    or_video_id
                ))

                existing_person = cursor.fetchone()
                if existing_person:
                    print(f"Duplicate entry found for person: {person_id} (age: {person_age}, gender: {person_gender})")
                    saved_paths.append(existing_person['person_origin_face'])  # 중복된 항목의 경로를 saved_paths에 추가
                    continue  # 중복이 있으면 건너뜁니다.

                # 이미지 파일 경로 설정
                person_image_dir = f'./extracted_images/{user_id}/filter_{filter_id}/{or_video_name}_clip/person_{person_id}/'
                if not os.path.exists(person_image_dir):
                    print(f"Directory not found: {person_image_dir}")
                    continue

                # 디렉토리 내의 이미지 파일 찾기
                face_files = [f for f in os.listdir(person_image_dir) if f.endswith('.jpg') or f.endswith('.png')]
                if not face_files:
                    print(f"No image files found in directory: {person_image_dir}")
                    continue

                # 첫 번째 이미지를 사용
                face_name = face_files[0]

                # 상대 경로로 저장
                face_image_relative_path = os.path.join(person_image_dir, face_name).replace("\\", "/")
                sql = """
                    INSERT INTO person (person_id, or_video_id, person_age, person_gender, person_upclothes, person_downclothes, person_face, person_origin_face, user_no, filter_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    person_id,
                    or_video_id,
                    person_age,
                    person_gender,
                    person_upclothes,
                    person_downclothes,
                    '',  # person_face
                    face_image_relative_path,  # person_origin_face
                    user_no,
                    filter_id
                ))
                saved_paths.append(face_image_relative_path)
        connection.commit()
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
    finally:
        connection.close()
    return saved_paths


# Person List DB 저장 (이미지 있을 때)
def save_to_db_with_image(person_info, or_video_id, user_id, user_no, filter_id, image_path):
    connection = get_db_connection()
    saved_paths = []
    # image_path 앞에 './'를 추가하고, 역슬래시를 슬래시로 변경
    image_path = f"./{image_path}".replace("\\", "/")
    try:
        with connection.cursor() as cursor:
            for person in person_info:
                or_video_name_result = get_or_video_name(or_video_id)
                if not or_video_name_result:
                    print(f"No or_video_name found for or_video_id: {or_video_id}")
                    continue

                or_video_name = os.path.splitext(or_video_name_result)[0]

                person_id = person['person_id']
                person_age = person['age']
                person_gender = person['gender']
                person_upclothes = person['upclothes']
                person_downclothes = person['downclothes']

                # 중복 체크
                check_sql = """
                    SELECT person_origin_face FROM person
                    WHERE person_age = %s
                    AND person_gender = %s
                    AND person_upclothes = %s
                    AND person_downclothes = %s
                    AND user_no = %s
                    AND filter_id = %s
                    AND or_video_id = %s
		    AND person_face = %s
                """
                cursor.execute(check_sql, (
                    person_age,
                    person_gender,
                    person_upclothes,
                    person_downclothes,
                    user_no,
                    filter_id,
                    or_video_id,
		    image_path
                ))

                existing_person = cursor.fetchone()
                if existing_person:
                    print(f"Duplicate entry found for person: {person_id} (age: {person_age}, gender: {person_gender})")
                    saved_paths.append(existing_person['person_origin_face'])  # 중복된 항목의 경로를 saved_paths에 추가
                    continue  # 중복이 있으면 건너뜁니다.

                # 이미지 파일 경로 설정
                person_image_dir = f'./extracted_images/{user_id}/filter_{filter_id}/{or_video_name}_clip/person_{person_id}/'
                if not os.path.exists(person_image_dir):
                    print(f"Directory not found: {person_image_dir}")
                    continue

                # 디렉토리 내의 이미지 파일 찾기
                face_files = [f for f in os.listdir(person_image_dir) if f.endswith('.jpg') or f.endswith('.png')]
                if not face_files:
                    print(f"No image files found in directory: {person_image_dir}")
                    continue

                # 첫 번째 이미지를 사용
                face_name = face_files[0]

                # 상대 경로로 저장
                face_image_relative_path = os.path.join(person_image_dir, face_name).replace("\\", "/")
                sql = """
                    INSERT INTO person (person_id, or_video_id, person_age, person_gender, person_upclothes, person_downclothes, person_face, person_origin_face, user_no, filter_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                cursor.execute(sql, (
                    person_id,
                    or_video_id,
                    person_age,
                    person_gender,
                    person_upclothes,
                    person_downclothes,
                    image_path,  # person_face
                    face_image_relative_path,  # person_origin_face
                    user_no,
                    filter_id
                ))
                saved_paths.append(face_image_relative_path)
        connection.commit()
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
    finally:
        connection.close()
    return saved_paths

# 클립 처리 함수
def clip_video(user_id, or_video_id, filter_id, video_names_for_clip_process, video_person_mapping, person_id):
    task_key = get_task_key(user_id, "clip", filter_id)
    if get_task_status(task_key) == 'running':
        print(f"Task {task_key} is already running")
        return

    set_task_status(task_key, 'running')
    print(f"Task {task_key} started")

    try:
        lock_file_path = f'/tmp/{user_id}_{filter_id}_clip.lock'
        with FileLock(lock_file_path):
            user_no = get_user_no(user_id)
            if user_no is not None:
                video_names_str = ','.join(video_names_for_clip_process)
                video_person_mapping_str = ','.join([f"{k}:{v}" for k, v in video_person_mapping.items()])
                
                process = subprocess.Popen(
                    ["python", "Clip.py", str(user_id), str(user_no), str(filter_id), video_names_str, video_person_mapping_str, person_id], 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate()
                    
                if process.returncode != 0:
                    print(f"Error occurred: {stderr.decode('utf-8')}")
                else:
                    print("클립 추출 성공")
                    connection = get_db_connection()
                    with connection.cursor() as cursor:
                        sql = "SELECT person_no FROM clip WHERE person_no IN (SELECT person_no FROM person WHERE or_video_id = %s)"
                        cursor.execute(sql, (or_video_id,))
                        person_nos = cursor.fetchall()
                        for person_no in person_nos:
                            update_person_face_from_clip(person_no['person_no'])
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    finally:
        set_task_status(task_key, 'completed')
        print(f"Task {task_key} completed")

# 트래킹 처리 함수 (이미지 없을 때)
def tracking_video(video_name, user_id, or_video_id, filter_id, saved_paths, image_name, face_flag = False):
    try:
        clip_folder_path = f'./extracted_images/{user_id}/filter_{filter_id}/{video_name}_clip/'
       # output.mp4 파일 경로 설정
        pro_video_name = f"{video_name}_output.mp4"
        pro_video_path = os.path.abspath(os.path.join(clip_folder_path, pro_video_name)).replace("\\", "/")
        # _clip 폴더와 output.mp4 파일이 존재하는지 확인
        if os.path.exists(clip_folder_path) and os.path.isdir(clip_folder_path) and os.path.exists(pro_video_path):
            print(f"{clip_folder_path} and {pro_video_path} exist, skipping subprocess")
            user_no = get_user_no(user_id)
            if user_no is not None:
                save_processed_video_info(video_name, user_id, user_no, or_video_id, filter_id)
        else:
            # face_flag를 사용하는 부분에서 str()로 변환
            if isinstance(face_flag, bool):
                face_flag = str(face_flag)  # 또는 필요하다면 다른 적절한 처리
            # Join the paths into a single string, separating each path with a comma or another delimitery
            paths_str = ','.join(saved_paths)
            print(f"{video_name} => Tracking  - - - paths : {paths_str}")
            process = subprocess.Popen(
                ["python", "Tracking.py", video_name, str(user_id), str(filter_id), paths_str, image_name, face_flag], 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(f"Error occurred: {stderr.decode('utf-8')}")
            else:
                print(f"{video_name} 트래킹 영상 추출 성공")
                user_no = get_user_no(user_id)
                if user_no is not None:
                    save_processed_video_info(video_name, user_id, user_no, or_video_id, filter_id)     

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 트래킹 콜백 처리 함수 (이미지 있을 때) 중복 작업을 방지
def tracking_video_with_image_callback(video_name, user_id, or_video_id, filter_id, image_paths, task_manager, person_id):
    task_key = get_task_key(user_id, video_name, filter_id)
    if get_task_status(task_key) == 'running':
        print(f"Task {task_key} is already running")
        return

    set_task_status(task_key, 'running')
    print(f"Tracking task {task_key} started")

    try:
        person_tracking_folder_path = f'./extracted_images/{user_id}/filter_{filter_id}/{video_name}_clip/person_{person_id}/'

        # 영상 파일 경로 생성
        output_video_path = os.path.join(person_tracking_folder_path, f"{video_name}_person_{person_id}_output.mp4")

        # {video_name}_person_{person_id}_output.mp4 영상이 존재하는지 확인
        if os.path.exists(output_video_path):
            print(f"{output_video_path} exists, skipping subprocess")
            task_manager.task_completed()
        else:
            paths_str = ','.join(image_paths)
            print(f"{video_name} => Tracking  - - - paths : {paths_str}")
            process = subprocess.Popen(
                ["python", "Tracking_with_image.py", video_name, str(user_id), str(filter_id), paths_str], 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(f"Error occurred: {stderr.decode('utf-8')}")
            else:
                print(f"{video_name} 트래킹 영상 추출 성공")
                task_manager.task_completed()
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    finally:
        set_task_status(task_key, 'completed')
        print(f"Tracking task {task_key} completed")

# 트래킹 영상 정보 저장 (이미지 없을 때)
def save_processed_video_info(video_name, user_id, user_no, or_video_id, filter_id):
    try:
        extracted_dir = f'./extracted_images/{user_id}/filter_{filter_id}/{video_name}_clip'
        if not os.path.exists(extracted_dir):
            print(f"No clip folder found for video {video_name}")
            return
        
        pro_video_name = f"{video_name}_output.mp4"
        pro_video_path = os.path.abspath(os.path.join(extracted_dir, pro_video_name)).replace("\\", "/")
        
        if not os.path.exists(pro_video_path):
            print(f"No processed video file found: {pro_video_path}")
            return
        
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                # 중복 체크 로직 추가
                sql_check = """
                    SELECT COUNT(*) as count FROM processed_video 
                    WHERE pro_video_name = %s AND or_video_id = %s AND filter_id = %s AND user_no = %s
                """
                cursor.execute(sql_check, (pro_video_name, or_video_id, filter_id, user_no))
                count = cursor.fetchone()['count']
                
                if count == 0:
                    sql = """
                        INSERT INTO processed_video (or_video_id, pro_video_name, pro_video_content, user_no, filter_id)
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (or_video_id, pro_video_name, pro_video_path, user_no, filter_id))
                    connection.commit()
                    print(f"Processed video info saved: {pro_video_name}")
                else:
                    print(f"Processed video already exists: {pro_video_name}")

        except pymysql.MySQLError as e:
            print(f"MySQL error occurred: {str(e)}")
        finally:
            connection.close()
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 얼굴 처리 함수 (이미지 없을 때)
def process_save_face_info_without_image(video_name, user_id, or_video_id, filter_id, clip_flag=True):
    try:
        # filter 정보 가져오기
        filter_info = get_filter_info(filter_id)
        if filter_info:
            filter_gender = filter_info['filter_gender']
            filter_age = filter_info['filter_age']
            filter_upclothes = filter_info['filter_upclothes']
            filter_downclothes = filter_info['filter_downclothes']
        else:
            print(f"No filter found for filter_id: {filter_id}")
            return
        
        # 'None' 값을 'none' 문자열로 변환
        filter_gender = 'none' if filter_gender is None else filter_gender
        filter_age = 'none' if filter_age is None else filter_age
        filter_upclothes = 'none' if filter_upclothes is None else filter_upclothes
        filter_downclothes = 'none' if filter_downclothes is None else filter_downclothes

        # save_face_info6.py 스크립트 호출 (백그라운드 실행)
        process = subprocess.Popen(
            ["python", "Save_info.py", video_name, str(user_id), str(filter_id), filter_gender, filter_age, filter_upclothes, filter_downclothes], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error occurred: {stderr.decode('utf-8')}")
        else:
            print(f"{video_name} Save_info.py 정보 추출 성공")
            # 예시 메모장 파일 경로
            info_file_path = f'./extracted_images/{user_id}/filter_{filter_id}/{video_name}_face_info.txt'

            # 파싱한 person 정보
            person_info = parse_info_file(info_file_path)

            # DB에 저장
            user_no = get_user_no(user_id)
            if user_no is not None:
                # 이미지 파일 경로 설정
                saved_paths = save_to_db(person_info, or_video_id, user_id, user_no, filter_id)
                print("person DB 저장")
                print("Saved image paths:", saved_paths)
            image_name = "none"
            tracking_video(video_name, user_id, or_video_id, filter_id, saved_paths, image_name, face_flag=False)
            print("pro_video db 저장")
            #if clip_flag:
                #clip_video(video_name, user_id, or_video_id)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 얼굴 처리 함수 (이미지 있을 때)
def process_save_face_info_with_image(video_name, user_id, or_video_id, filter_id, image_path, image_name, clip_flag=True, face_flag = False):
    try:
        # filter 정보 가져오기
        filter_info = get_filter_info(filter_id)
        if filter_info:
            filter_gender = filter_info['filter_gender']
            filter_age = filter_info['filter_age']
            filter_upclothes = filter_info['filter_upclothes']
            filter_downclothes = filter_info['filter_downclothes']
        else:
            print(f"No filter found for filter_id: {filter_id}")
            return
        
        # 'None' 값을 'none' 문자열로 변환
        filter_gender = 'none' if filter_gender is None else filter_gender
        filter_age = 'none' if filter_age is None else filter_age
        filter_upclothes = 'none' if filter_upclothes is None else filter_upclothes
        filter_downclothes = 'none' if filter_downclothes is None else filter_downclothes

        # save_face_info6.py 스크립트 호출 (백그라운드 실행)
        process = subprocess.Popen(
            ["python", "Save_info_with_image.py", video_name, str(user_id), str(filter_id), filter_gender, filter_age, filter_upclothes, filter_downclothes, image_path], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error occurred: {stderr.decode('utf-8')}")
        else:
            print(f"{video_name} Save_info.py 정보 추출 성공")
            # 예시 메모장 파일 경로
            info_file_path = f'./extracted_images/{user_id}/filter_{filter_id}/{video_name}_face_info.txt'

            # 파싱한 person 정보
            person_info = parse_info_file(info_file_path)

            # DB에 저장
            user_no = get_user_no(user_id)
            if user_no is not None:
                # 이미지 파일 경로 설정
                saved_paths = save_to_db_with_image(person_info, or_video_id, user_id, user_no, filter_id, image_path)
                print("person DB 저장")
                print("Saved image paths:", saved_paths)
            
            tracking_video(video_name, user_id, or_video_id, filter_id, saved_paths, image_name, face_flag)
            print("pro_video db 저장")

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 벡터 추출 안되는 이미지 삭제 후 (이미지 없을 때 정보저장 실행)
def process_image_filter_without_image(video_name, user_id, filter_id, clip_flag=True):
    try:
        # Main_image2.py 스크립트 호출 (백그라운드 실행)
        process = subprocess.Popen(
            ["python", "Delete_Not_Extract_Face.py", video_name, str(user_id)], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error occurred: {stderr.decode('utf-8')}")
        else:
            print("Delete_strange_image.py 이미지 필터링 성공")
            # 얼굴정보추출 성공 후 save_face_info6.py 실행
            video_path = os.path.join('uploaded_videos', user_id, video_name + ".mp4").replace("\\", "/")
            or_video_id = get_or_video_id_by_path(video_path)
            if or_video_id is not None:
                process_save_face_info_without_image(video_name, user_id, or_video_id, filter_id, clip_flag)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 벡터 추출 안되는 이미지 삭제 후 (이미지 있을 때 정보저장 실행)
def process_image_filter_with_image(video_name, user_id, filter_id, image_path, image_name, clip_flag=True, face_flag = False):
    try:
        # Main_image2.py 스크립트 호출 (백그라운드 실행)
        process = subprocess.Popen(
            ["python", "Delete_Not_Extract_Face.py", video_name, str(user_id)], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error occurred: {stderr.decode('utf-8')}")
        else:
            print(f"Delete_strange_image.py {video_name}_face 이미지 필터링 성공")
            # 얼굴정보추출 성공 후 save_face_info6.py 실행
            video_path = os.path.join('uploaded_videos', user_id, video_name + ".mp4").replace("\\", "/")
            or_video_id = get_or_video_id_by_path(video_path)
            if or_video_id is not None:
                process_save_face_info_with_image(video_name, user_id, or_video_id, filter_id, image_path, image_name, clip_flag, face_flag)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 비디오 처리 함수 (이미지 없을 때)
def process_video_without_images(video_name, user_id, filter_id, clip_flag=True):
    try:
        # 디렉토리 경로 설정
        face_folder_path = f"./extracted_images/{user_id}/{video_name}_face"
        
        # {video_name}_face 폴더가 존재하는지 확인
        if os.path.exists(face_folder_path) and os.path.isdir(face_folder_path):
            print(f"{face_folder_path} exists, skipping subprocess")
            process_image_filter_without_image(video_name, user_id, filter_id, clip_flag)
        else:
            # Main_test.py 스크립트 호출 (백그라운드 실행)
            process = subprocess.Popen(
                ["python", "Main.py", video_name, str(user_id)], 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                print(f"Error occurred: {stderr.decode('utf-8')}")
            else:
                print("Main.py 얼굴정보추출 성공")
                process_image_filter_without_image(video_name, user_id, filter_id, clip_flag)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 비디오 처리 함수 (이미지 있을 때)
def process_video_with_images(video_name, user_id, filter_id, image_path, image_name, clip_flag=True, face_flag = False):
    try:
        # 디렉토리 경로 설정
        face_folder_path = f"./extracted_images/{user_id}/{video_name}_face"
        
        # {video_name}_face 폴더가 존재하는지 확인
        if os.path.exists(face_folder_path) and os.path.isdir(face_folder_path):
            print(f"{face_folder_path} exists, skipping subprocess")
            process_image_filter_with_image(video_name, user_id, filter_id, image_path, image_name, clip_flag, face_flag)
        # Main_image2.py 스크립트 호출 (백그라운드 실행)
        else:
            process = subprocess.Popen(
                ["python", "Main.py", video_name, str(user_id)], 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                print(f"Error occurred: {stderr.decode('utf-8')}")
            else:
                print("Main.py 얼굴정보추출 성공")
                process_image_filter_with_image(video_name, user_id, filter_id, image_path, image_name, clip_flag, face_flag)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 비디오 분류후 처리 메서드 실행
def process_video(video_name, user_id, filter_id, image_path, image_name, clip_flag):
    video_base_name = os.path.splitext(video_name)[0]
    if image_path:
        face_flag = True
        process_video_with_images(video_base_name, user_id, filter_id, image_path, image_name, clip_flag, face_flag)
    else:
        process_video_without_images(video_base_name, user_id, filter_id, clip_flag)

#스레드 제한 및 작업상태 추적 메서드
def process_videos(video_names, user_id, filter_id, image_path, clip_flag, video_filter_map, image_name):
    processed_videos = set()  # 집합 대신 리스트로 변경
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for video_name in video_names:
            if video_name not in video_filter_map:
                if video_name not in processed_videos:
                    future = executor.submit(process_video, video_name, user_id, filter_id, image_path, image_name, clip_flag)
                    futures[future] = video_name
                    processed_videos.add(video_name)  # 집합 대신 리스트로 변경
            else:
                print(f"{video_name}_처리영상 존재")

        for future in concurrent.futures.as_completed(futures):
            video_name = futures[future]
            try:
                future.result()
                print(f"{video_name} 처리 완료")
            except Exception as e:
                print(f"{video_name} 처리 중 에러 발생: {e}")

# ffmpeg를 사용하여 비디오를 H.264 코덱으로 재인코딩.
def reencode_video(input_path, output_path):
    command = [
        'ffmpeg', '-i', input_path, '-c:v', 'libx264', '-b:v', '2000k',
        '-c:a', 'aac', '-b:a', '128k', output_path
    ]
    subprocess.run(command, check=True)

# 1. 파일 업로드 엔드포인트 (Post)
@app.route('/upload_file', methods=['POST'])
def upload_file():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data received"}), 400
        if isinstance(data, str):
            return jsonify({"status": "error", "message": "Invalid JSON data format"}), 400

        user_data = data.get('user_data', {})
        user_id = user_data.get('user_id', '')
        user_no = get_user_no(user_id)

        filter_data = data.get('filter_data', {})
        age = filter_data.get('age', None)
        gender = filter_data.get('gender', None)
        uptype = filter_data.get('uptype', None)
        downtype = filter_data.get('downtype', None)

        bundle_data = data.get('bundle_data', {})
        bundle = bundle_data.get('bundle_name', None)
        
        clip_flag = data.get('clip_flag', 'true').lower() != 'false'

        user_video_path = os.path.join(VIDEO_SAVE_PATH, str(user_id))
        user_image_path = os.path.join(IMAGE_SAVE_PATH, str(user_id))
        os.makedirs(user_video_path, exist_ok=True)
        os.makedirs(user_image_path, exist_ok=True)

        connection = get_db_connection()
        video_names = []
        total_video_length = 0  
        filter_id = None

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

        with connection.cursor() as cursor:
            # Check if a filter with the same details already exists
            filter_check_sql = """
                SELECT filter_id FROM filter
                WHERE (filter_gender IS NULL OR filter_gender = %s) 
                AND (filter_age IS NULL OR filter_age = %s)
                AND (filter_upclothes IS NULL OR filter_upclothes = %s)
                AND (filter_downclothes IS NULL OR filter_downclothes = %s)
                AND (bundle_name IS NULL OR bundle_name = %s)
                AND (filter_image IS NULL OR filter_image = %s)
                AND (user_no IS NULL OR user_no = %s)
            """
            cursor.execute(filter_check_sql, (
                gender if gender is not None else None,
                age if age is not None else None,
                uptype if uptype is not None else None,
                downtype if downtype is not None else None,
                bundle if bundle is not None else None,
                image_path if image_path is not None else None,
                user_no if user_no is not None else None
            ))
            existing_filter = cursor.fetchone()

            if existing_filter:
                filter_id = existing_filter['filter_id']
                print(f"Using existing filter ID: {filter_id}")
            else:
                # If not exists, create a new filter
                filter_sql = """
                    INSERT INTO filter (filter_gender, filter_age, filter_upclothes, filter_downclothes, bundle_name, filter_image, user_no)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(filter_sql, (gender, age, uptype, downtype, bundle, image_path, user_no))
                filter_id = cursor.lastrowid
                print("New filter created")
                print(f"New filter ID : {filter_id}")

            video_data = data.get('video_data', [])
            for video in video_data:
                video_name = video.get('video_name', '')
                video_content_base64 = video.get('video_content', '')
                start_time = ''
                cam_name = video.get('cam_name', '')
                address = video.get('address', '')

                if video_name and video_content_base64:
                    video_content = base64.b64decode(video_content_base64)
                    video_path = os.path.join(user_video_path, video_name).replace("\\", "/")
                    with open(video_path, 'wb') as video_file:
                        video_file.write(video_content)

                    # 비디오 파일 길이 계산 및 추가
                    video_length = get_video_length(video_path)
                    total_video_length += video_length

                    # 비디오 시작 시간 가져오기
                    if not start_time:  # 이미 제공된 시작 시간이 없다면
                        start_time = get_video_metadata(video_path)

                    start_time = convert_timestamp_format(start_time)

                    # cam_name과 다른 필요한 정보를 이용해 cam_num 조회
                    sql = """
                        SELECT cam_num 
                        FROM camera 
                        WHERE cam_name = %s 
                        AND map_num IN (SELECT map_num FROM map WHERE user_no = (SELECT user_no FROM user WHERE user_id = %s AND address = %s))
                    """
                    cursor.execute(sql, (cam_name, user_id, address))
                    cam_results = cursor.fetchall()
                    if not cam_results:
                        print(f"No cam_num found for cam_name: {cam_name} and user_id: {user_id}")
                        continue

                    for cam_result in cam_results:
                        cam_num = cam_result['cam_num']

                        # 중복 체크
                        sql_check = """
                            SELECT COUNT(*) as count FROM origin_video
                            WHERE or_video_name = %s AND start_time = %s AND cam_num = %s
                        """
                        cursor.execute(sql_check, (video_name, start_time, cam_num))
                        count = cursor.fetchone()['count']

                        if count == 0:
                            # origin_video에 데이터 삽입
                            sql = """
                                INSERT INTO origin_video (or_video_name, or_video_content, start_time, cam_num)
                                VALUES (%s, %s, %s, %s)
                            """
                            cursor.execute(sql, (video_name, video_path, start_time, cam_num))
                            video_names.append(video_name)
                        else:
                            video_names.append(video_name)
                            print(f"Video already exists: {video_name} at {start_time} for cam_num {cam_num}")

            video_filter_map = {}
            for video_name in video_names:
                filter_ids = get_filter_ids_by_video_name_and_user(video_name, user_id)
                if filter_ids:
                    valid_filter_ids = []
                    for fid in filter_ids:
                        filter_info = get_filter_info(fid)                                                                                                  #filter_info['filter_downclothes'] == downtype and
                        if filter_info and filter_info['filter_gender'] == gender and filter_info['filter_age'] == age and filter_info['filter_upclothes'] == uptype and filter_info['filter_downclothes'] == downtype and filter_info['bundle_name'] == bundle:
                            valid_filter_ids.append(fid)
                    if valid_filter_ids:
                        video_filter_map[video_name] = valid_filter_ids

            print(video_filter_map)  # For debugging
            print(f"Total video length: {total_video_length} seconds")  # 총 비디오 길이 출력

            connection.commit()
            connection.close()

            response_data = {
                    "status": "success",
                    "message": "Data received and processed successfully",
                    "filter_id": filter_id,
                }

            response = jsonify(response_data)
            response.status_code = 200


            # image_name에서 .mp4 확장자를 제거
            image_name = image_name.replace('.jpg', '')
            # Process videos not in video_filter_map
            process_videos(video_names, user_id, filter_id, image_path, clip_flag, video_filter_map, image_name)

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

# 2.회원가입 엔드포인트 (Post)
@app.route('/receive_data', methods=['POST'])
def receive_data():
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
                cursor.execute(check_sql, (user['ID'],))
                result = cursor.fetchone()
                
                print("Check result:", result)  # 디버깅 메시지 추가
                if result is not None and result['user_id'] == user['ID']:
                    print(f"User ID {user['ID']} already exists")  # 디버깅 메시지 추가
                    return jsonify({"error": "이미 존재하는 ID입니다"}), 409

                # User 테이블에 삽입
                user_sql = "INSERT INTO user (user_id) VALUES (%s)"
                cursor.execute(user_sql, (user['ID'],))
                print("Inserted into user table")  # 디버깅 메시지 추가
                
                # 방금 삽입한 user_no 가져오기
                user_no = cursor.lastrowid
                print("Inserted user_no:", user_no)  # 디버깅 메시지 추가

                # Password 테이블에 삽입
                password_sql = "INSERT INTO password (user_no, password) VALUES (%s, %s)"
                cursor.execute(password_sql, (user_no, user['PW']))
                print("Inserted into password table")  # 디버깅 메시지 추가

                # Profile 테이블에 삽입
                profile_sql = "INSERT INTO profile (user_no, user_name) VALUES (%s, %s)"
                cursor.execute(profile_sql, (user_no, user['Name']))
                print("Inserted into profile table")  # 디버깅 메시지 추가
            
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
    
# 3. 로그인 엔드포인트 (Post)
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    print("Login attempt:", data)
    
    if data and 'ID' in data and 'PW' in data:
        connection = get_db_connection()
        cursor = connection.cursor()  # dictionary 인자 제거

        try:
            login_sql = """
                SELECT u.user_no, u.user_id, p.password
                FROM user u
                JOIN password p ON u.user_no = p.user_no
                WHERE u.user_id = %s AND p.password = %s
            """
            cursor.execute(login_sql, (data['ID'], data['PW']))
            result = cursor.fetchone()

            if result is not None:
                user_no = result['user_no']
                user_id = result['user_id']

                profile_sql = """
                    SELECT profile.user_name
                    FROM profile
                    WHERE profile.user_no = %s;
                """
                cursor.execute(profile_sql, (user_no,))
                profile_result = cursor.fetchone()
                user_name = profile_result['user_name'] if profile_result else "Unknown"

                map_camera_provideo_sql = """
                     SELECT 
                        m.address, 
                        m.map_latitude, 
                        m.map_longitude, 
                        c.cam_name, 
                        c.cam_latitude, 
                        c.cam_longitude, 
                        p.person_id,
                        f.filter_id,
                        f.filter_gender,
                        f.filter_age,
                        f.filter_upclothes,
                        f.filter_downclothes,
                        f.bundle_name,
                        pv.pro_video_name
                    FROM 
                        map m
                    JOIN 
                        camera c ON m.map_num = c.map_num
                    LEFT JOIN 
                        origin_video orv ON c.cam_num = orv.cam_num
                    LEFT JOIN 
                        person p ON p.or_video_id = orv.or_video_id
                    LEFT JOIN 
                        filter f ON f.filter_id = p.filter_id
					LEFT JOIN
						processed_video pv ON f.filter_id = pv.filter_id
                    WHERE 
                        m.user_no = %s;
                """
                cursor.execute(map_camera_provideo_sql, (user_no,))
                map_camera_provideo_result = cursor.fetchall()

                camname_sql = """
                    SELECT 
                        camera.cam_name
                    FROM 
                        user 
                    JOIN 
                        processed_video p_video ON user.user_no = p_video.user_no
                    JOIN 
                        origin_video o_video ON p_video.or_video_id = o_video.or_video_id
                    JOIN 
                        camera ON o_video.cam_num = camera.cam_num
                    WHERE 
                        user.user_no = %s
                """
                cursor.execute(camname_sql, (user_no,))
                camname_result = cursor.fetchall()

                response_data = {
                    "message": "Login successful",
                    "user_id": user_id,
                    "user_name": user_name,
                    "map_camera_provideo_info": map_camera_provideo_result,
                    "camname_info": camname_result,
                }

                return jsonify(response_data), 200
            else:
                return jsonify({"error": "Invalid ID or password"}), 401
        except Exception as e:
            print(f"Exception: {str(e)}")
            return jsonify({"error": str(e)}), 500
        finally:
            cursor.close()
            connection.close()
    else:
        return jsonify({"error": "No data received or invalid format"}), 400
    
# 4.지도 주소 엔드포인트 (Post)
@app.route('/upload_maps', methods=['POST'])
def upload_map():
    try:
        data = request.get_json()
        if not data:
            print("No JSON data received")
            return jsonify({"error": "No JSON data received"}), 400

        print(f"Received data: {data}")

        connection = get_db_connection()
        print("Database connection established")
        with connection.cursor() as cursor:
            user_id = data.get('user_id')
            address = data.get('address')
            map_latitude = data.get('map_latitude')
            map_longitude = data.get('map_longitude')

            if user_id and map_latitude is not None and map_longitude is not None and address:
                # user_id를 이용하여 user_no 조회
                cursor.execute("SELECT user_no FROM user WHERE user_id = %s", (user_id))
                result = cursor.fetchone()

                if not result:
                    print(f"user_id {user_id} not found")
                    return jsonify({"error": "user_id not found"}), 404

                user_no = result['user_no']
                print(f"Found user_no: {user_no}")

                # map 테이블에 데이터 삽입
                sql = """
                    INSERT INTO map (address, map_latitude, map_longitude, user_no)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(sql, (address, map_latitude, map_longitude, user_no))
                print(f"Inserted: {address}, {map_latitude}, {map_longitude}, {user_no}")

                connection.commit()
                print("Transaction committed")

                cursor.execute("SELECT LAST_INSERT_ID() as map_num")
                result = cursor.fetchone()
                map_num = result['map_num']
                print(f"Retrieved map_num: {map_num}")

            else:
                print("Invalid data:", data)
                return jsonify({"error": "Invalid data format received"}), 400

        connection.close()
        print("Database connection closed")
        return jsonify({"message": "Map saved successfully", "map_num": map_num}), 200

    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return jsonify({"error": f"MySQL error occurred: {str(e)}"}), 500
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# 5.지도 마커 위치 엔드포인트 (Post)
@app.route('/upload_cams', methods=['POST'])
def upload_cameras():
    try:
        data = request.get_json()
        if not data:
            print("No JSON data received")
            return jsonify({"error": "No JSON data received"}), 400

        if not isinstance(data, list):
            print("Invalid data format received")
            return jsonify({"error": "Invalid data format received"}), 400

        print(f"Received data: {data}")

        connection = get_db_connection()
        print("Database connection established")
        with connection.cursor() as cursor:
            cursor.execute("SELECT map_num FROM map ORDER BY map_num DESC LIMIT 1")
            result = cursor.fetchone()
            if result:
                map_num = result['map_num']
                print(f"Retrieved latest map_num: {map_num}")
            else:
                print("No map records found")
                return jsonify({"error": "No map records found"}), 400

            existing_names = set()
            for camera in data:
                original_name = camera.get('name')
                cam_name = original_name
                count = 1
                while cam_name in existing_names:
                    cam_name = f"{original_name}_{count}"
                    count += 1
                existing_names.add(cam_name)

                cam_latitude = camera.get('latitude')
                cam_longitude = camera.get('longitude')

                if cam_name and cam_latitude and cam_longitude:
                    sql = """
                        INSERT INTO camera (cam_name, map_num, cam_latitude, cam_longitude)
                        VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(sql, (cam_name, map_num, cam_latitude, cam_longitude))
                    print(f"Inserted: {cam_name}, {map_num}, {cam_latitude}, {cam_longitude}")
                else:
                    print("Invalid camera data:", camera)
                    return jsonify({"error": "Invalid camera data format received"}), 400

            connection.commit()
            print("Transaction committed")

        connection.close()
        print("Database connection closed")
        return jsonify({"message": "Cameras saved successfully"}), 200

    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return jsonify({"error": f"MySQL error occurred: {str(e)}"}), 500
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# 6.지도 업데이트 엔드포인트 (GET) # 쓰레기
@app.route('/update_maps', methods=['GET'])
def upload_maps():
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # user_id를 기반으로 user_no를 찾기
        user_sql = "SELECT user_no FROM user WHERE user_id = %s"
        cursor.execute(user_sql, (user_id,))
        user_result = cursor.fetchone()

        if user_result is None:
            return jsonify({"error": "User not found"}), 404

        user_no = user_result['user_no']

        # user_no를 기반으로 map_camera_info를 가져오기
        map_camera_sql = """
            SELECT 
                m.address, 
                m.map_latitude, 
                m.map_longitude, 
                c.cam_name, 
                c.cam_latitude, 
                c.cam_longitude
            FROM map m
            LEFT JOIN camera c ON m.map_num = c.map_num
            WHERE m.user_no = %s
        """
        cursor.execute(map_camera_sql, (user_no,))
        map_camera_result = cursor.fetchall()

        map_camera_dict = [dict(row) for row in map_camera_result]

        return jsonify({"map_camera_info": map_camera_dict}), 200

    except Exception as e:
        print(f"Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()

# 7.ProVideo 업데이트 엔드포인트 (GET)
@app.route('/update_pro_video', methods=['GET'])
def update_pro_video():
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # user_id를 기반으로 user_no를 찾기
        user_sql = "SELECT user_no FROM user WHERE user_id = %s"
        cursor.execute(user_sql, (user_id,))
        user_result = cursor.fetchone()

        if user_result is None:
            return jsonify({"error": "User not found"}), 404

        user_no = user_result['user_no']

        # map, ProVideo, bundle 정보 가져오기
        map_camera_provideo_sql = """
            SELECT 
                m.address, 
                m.map_latitude, 
                m.map_longitude, 
                c.cam_name, 
                c.cam_latitude, 
                c.cam_longitude, 
                p.person_id,
                f.filter_id,
                f.filter_gender,
                f.filter_age,
                f.filter_upclothes,
                f.filter_downclothes,
                f.bundle_name,
                pv.pro_video_name
            FROM 
                map m
            JOIN 
                camera c ON m.map_num = c.map_num
            LEFT JOIN 
                origin_video orv ON c.cam_num = orv.cam_num
            LEFT JOIN 
                person p ON p.or_video_id = orv.or_video_id
            LEFT JOIN 
                filter f ON f.filter_id = p.filter_id
            LEFT JOIN
                processed_video pv ON f.filter_id = pv.filter_id
            WHERE 
                m.user_no = %s;
        """
        cursor.execute(map_camera_provideo_sql, (user_no,))
        map_camera_provideo_result = cursor.fetchall()

        map_camera_provideo_dict = [dict(row) for row in map_camera_provideo_result] if map_camera_provideo_result else []

        return jsonify({"map_camera_provideo_info": map_camera_provideo_dict}), 200

    except Exception as e:
        print(f"Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()

# 8. Person 정보 가져오기 엔드포인트 (GET)
@app.route('/select_person', methods=['GET'])
def select_person():
    filter_id = request.args.get('filter_id')
    person_id = request.args.get('person_id')
    print(f"{filter_id}")

    if not filter_id:
        return jsonify({"error": "Filter Id is required"}), 400

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # filter_id를 기반으로 Person을 가져오기
        person_sql = """
            SELECT 
                p.person_id, 
                p.person_age, 
                p.person_gender, 
                p.person_upclothes, 
                p.person_downclothes, 
                p.person_origin_face
            FROM person p
            JOIN processed_video pv ON p.or_video_id = pv.or_video_id
            WHERE 
                p.filter_id = %s and p.person_id = %s
            """
        cursor.execute(person_sql, (filter_id, person_id))
        person_result = cursor.fetchall()
        print(person_result)
        person_dict = []
        for row in person_result:
            person_info = dict(row)
            face_path = person_info['person_origin_face']

            # 이미지 파일을 Base64로 인코딩
            if os.path.exists(face_path):
                with open(face_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    person_info['person_origin_face'] = encoded_string
            else:
                person_info['person_origin_face'] = None  # 파일이 없을 경우 None으로 설정

            person_dict.append(person_info)

        # filter_id와 함께 응답하기
        return jsonify({"filter_id": filter_id, "person_info": person_dict}), 200

    except Exception as e:
        print(f"Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()

# 9. Map 경로 관련 엔드포인트 (GET)
@app.route('/map_cal', methods=['GET'])
def map_cal():
    person_id = request.args.get('person_id')

    if not person_id:
        return jsonify({"error": "Filter Id is required"}), 400

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        clip_cam_sql = """
            SELECT
                c.clip_id, c.clip_time, c.clip_video, cam.cam_name, cam.cam_latitude, cam.cam_longitude, m.address
            FROM clip c
            JOIN person p ON c.person_no = p.person_no
            JOIN origin_video o ON p.or_video_id = o.or_video_id
            JOIN camera cam ON o.cam_num = cam.cam_num
            JOIN map m ON cam.map_num = m.map_num
            WHERE p.person_id = %s
            """
        
        cursor.execute(clip_cam_sql, (person_id,))
        clip_cam_result = cursor.fetchall()
        
        order_data = []
        location_data = []
        address_data = []

        for row in clip_cam_result:
            order_data.append({
                "clip_id": row['clip_id'],
                "clip_time": row['clip_time'].strftime("%Y-%m-%d %H:%M:%S"),
                "cam_name": row['cam_name']
            })
            location_data.append({
                "cam_name": row['cam_name'],
                "cam_latitude": float(row['cam_latitude']),
                "cam_longitude": float(row['cam_longitude']),
                "clip_time": row['clip_time'].strftime("%Y-%m-%d %H:%M:%S")
            })
            address_data.append({
                 "address": row['address']
            })
        
        sorted_order_data = sorted(order_data, key=lambda x: x['clip_time'])
        cam_name_order = [entry['cam_name'] for entry in sorted_order_data]
        cam_name_order_str = '/'.join(cam_name_order)

        coordinates = {entry['cam_name']: (entry['cam_latitude'], entry['cam_longitude']) for entry in location_data}
        distance = total_distance(coordinates, cam_name_order)

        start_time = sorted_order_data[0]['clip_time']
        end_time = sorted_order_data[-1]['clip_time']

        last_camera_name = sorted_order_data[-1]['cam_name']
        last_camera_coords = coordinates[last_camera_name]
        last_cam_latitude = last_camera_coords[0]
        last_cam_longitude = last_camera_coords[1]

        current_time = datetime.now()
        radius = calculate_radius(start_time, end_time, distance, current_time)

        response_data = {
            "order_data": cam_name_order_str,
            "radius": radius,
            "last_camera_name": last_camera_name,
            "last_camera_latitude": last_cam_latitude,
            "last_camera_longitude": last_cam_longitude,
            "address": address_data,
            "cam": location_data
        }

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()
        
# 10. Pro_video 동영상 스트리밍 엔드포인트 (GET)
@app.route('/stream_video', methods=['GET'])
def stream_video():
    user_id = request.args.get('user_id')
    filter_id = request.args.get('filter_id')
    print("user_id:", user_id)  # 디버깅 메시지 추가
    print("filter_id:", filter_id)  # 디버깅 메시지 추가

    video_paths = get_video_path(user_id, filter_id)
    if not video_paths or not isinstance(video_paths, list):
        return jsonify({"error": "Invalid user_id or pro_video_name"}), 404

    video_urls = []

    for video_path in video_paths:
        if not os.path.exists(video_path):
            return jsonify({"error": f"Video file not found: {video_path}"}), 404

        reencoded_path = f"{video_path.rsplit('.', 1)[0]}_h264.mp4"
        print(f"Original video path: {video_path}")  # 디버깅 메시지 추가
        print(f"Reencoded video path: {reencoded_path}")  # 디버깅 메시지 추가

        if not os.path.exists(reencoded_path):
            try:
                reencode_video(video_path, reencoded_path)
                if not os.path.exists(reencoded_path):
                    print(f"Reencoded file does not exist after re-encoding: {reencoded_path}")
                    return jsonify({"error": "Failed to save re-encoded video"}), 500
            except subprocess.CalledProcessError as e:
                print(f"Error during re-encoding: {str(e)}")
                return jsonify({"error": "Error during re-encoding"}), 500

        video_url = url_for('get_video', filename=os.path.basename(reencoded_path), user_id=user_id, filter_id=filter_id, _external=True)
        video_urls.append(video_url)

    return jsonify({"video_urls": video_urls, "user_id" : user_id, "filter_id" : filter_id}), 200

@app.route('/videos/<filename>')
def get_video(filename):
    user_id = request.args.get('user_id')
    filter_id = request.args.get('filter_id')
    video_name = filename.split('_')[0]
    file_path = os.path.join(f'./extracted_images/{user_id}/filter_{filter_id}/{video_name}_clip', filename)  # 실제 경로로 수정 필요
    if not os.path.exists(file_path):
        return jsonify({"error": "Video file not found"}), 404

    def generate():
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(1024*1024)
                if not chunk:
                    break
                yield chunk

    range_header = request.headers.get('Range', None)
    if not range_header:
        return Response(generate(), mimetype='video/mp4')

    size = os.path.getsize(file_path)
    byte1, byte2 = 0, None

    if '-' in range_header:
        byte1, byte2 = range_header.replace('bytes=', '').split('-')
    
    byte1 = int(byte1)
    if byte2:
        byte2 = int(byte2)
        length = byte2 - byte1 + 1
    else:
        length = size - byte1

    def generate_range():
        with open(file_path, 'rb') as f:
            f.seek(byte1)
            remaining_length = length
            while remaining_length > 0:
                chunk = f.read(min(1024*1024, remaining_length))
                if not chunk:
                    break
                remaining_length -= len(chunk)
                yield chunk

    rv = Response(generate_range(), 206, mimetype='video/mp4')
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{size}')
    return rv

# 11. Clip 생성을 위한 전처리 엔드포인트 (GET) 
@app.route('/get_Person_to_clip', methods=['GET'])
def get_Person_to_clip():
    user_id = request.args.get('user_id')
    person_id = request.args.get('person_id')
    filter_id = request.args.get('filter_id')

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    if not person_id:
        return jsonify({"error": "Person ID is required"}), 400
    
    if not filter_id:
        return jsonify({"error": "Filter_ID is required"}), 400
    
    print(f"clip_process_info : userid : {user_id} personid : {person_id} filterid : {filter_id}")

    task_key = f'{user_id}_{person_id}_{filter_id}'
    if get_task_status(task_key) == 'running':
        print(f"Task {task_key} is already running")
        return jsonify({"message": "Task is already running"}), 200

    connection = get_db_connection()
    cursor = connection.cursor()
    clip_info = []
    try:
        set_task_status(task_key, 'running')
        print(f"Task {task_key} started")
        user_no = get_user_no(user_id)
        person_nos2 = get_person_nos2(person_id, user_no)
        person_nos = get_person_nos(person_id, user_no, filter_id)
        if not person_nos:
            return jsonify({"error": "Person not found"}), 404

        or_video_ids = [get_or_video_id_by_person_no(person_no) for person_no in person_nos]
        video_names = [get_or_video_name(or_video_id) for or_video_id in or_video_ids]

        video_names_for_clip_process = []
        video_person_mapping = {}
        for i, or_video_id in enumerate(or_video_ids):
            video_names_for_clip_process.extend(get_vidoe_names_by_or_video_id_and_user_no_and_filter_id(or_video_id, user_no, filter_id))
            video_person_mapping[video_names[i]] = person_nos[i]

        lock_file_path = f'/tmp/{user_id}_{video_names[0]}_{person_id}.lock'
        with FileLock(lock_file_path):
            pro_videos = []
            pro_video_names = []
            pro_videos = get_pro_video_ids_by_filter_id(filter_id)
            pro_video_names = get_pro_video_names_by_ids(pro_videos)

            # or_video_ids 리스트에 해당하는 pro_videos 필터링
            pro_video_id_to_or_video_id = get_or_video_ids_by_pro_video_ids(pro_videos)
            filtered_pro_videos = [pro_video for pro_video, or_video in pro_video_id_to_or_video_id.items() if or_video in or_video_ids]
            filtered_pro_video_names = [name for name, pro_video in zip(pro_video_names, pro_videos) if pro_video in filtered_pro_videos]

            missing_videos = []
            for name in filtered_pro_video_names:
                if not does_video_file_exist(user_id, name, person_id, filter_id):
                    missing_videos.append(name)

            clip_count = 0
            for person_no in person_nos:
                clip_sql = """
                    SELECT COUNT(*) as count FROM clip WHERE person_no = %s
                """
                cursor.execute(clip_sql, (person_no,))
                clip_count += cursor.fetchone()['count']

            # 트래킹 영상이 존재하지 않거나 clip_count가 0인 경우 클립 생성
            if missing_videos or clip_count == 0:
                for person_no in person_nos2:
                    for video_name in missing_videos:
                        clip_sql = """
                            SELECT c.clip_video, c.clip_time, cam.cam_name
                            FROM clip c
                            JOIN camera cam ON c.cam_num = cam.cam_num
                            WHERE c.person_no = %s
                            AND c.clip_video LIKE %s
                        """
                        cursor.execute(clip_sql, (person_no, f"%{video_name}_person_{person_id}_output%"))
                        clip_result = cursor.fetchall()
                        
                        if clip_result:
                            clip_info.extend([dict(row) for row in clip_result])
                if clip_info:
                    return jsonify({"clip_info": clip_info}), 200
                else:
                    image_dir = f'./extracted_images/{user_id}/filter_{filter_id}/{video_names[0]}_clip/person_{person_id}/'
                    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
                    if not image_files:
                        return jsonify({"error": "No image files found to create tracking video"}), 404

                    image_path = os.path.join(image_dir, image_files[0]).replace("\\", "/")

                    
                    if clip_count == 0:
                        # Clip Process if clip_count is 0
                        task_manager = TrackingTaskManager2(video_names_for_clip_process, lambda: clip_video(user_id, or_video_ids[0], filter_id, video_names_for_clip_process, video_person_mapping, person_id))
                    else:
                        # Track if missing_videos 
                        task_manager = TrackingTaskManager(missing_videos, lambda: clip_video(user_id, or_video_ids[0], filter_id, video_names_for_clip_process, video_person_mapping, person_id))
                        
                    def tracking_callback(video_name, image_path):
                        tracking_video_with_image_callback(video_name, user_id, or_video_ids[0], filter_id, [image_path], task_manager, person_id)

                    def create_tracking_videos():
                        futures = []
                        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                            for name in video_names_for_clip_process:  # missing_videos가 아닌 video_names_for_clip_process 사용
                                # 각 비디오마다 올바른 이미지 경로를 가져오기
                                image_dir = f'./extracted_images/{user_id}/filter_{filter_id}/{name}_clip/person_{person_id}/'

                                if not os.path.exists(image_dir):
                                    print(f"Directory does not exist: {image_dir}")
                                    if name in video_names_for_clip_process:
                                        video_names_for_clip_process.remove(name)
                                    if name in video_person_mapping:
                                        del video_person_mapping[name]
                                    continue

                                image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
                                if not image_files:
                                    print(f"No image files found for {name}")
                                    continue

                                image_path = os.path.join(image_dir, image_files[0]).replace("\\", "/")
                                futures.append(executor.submit(tracking_callback, name, image_path))

                            # 모든 트래킹 작업이 완료될 때까지 기다립니다.
                            for future in concurrent.futures.as_completed(futures):
                                try:
                                    future.result()
                                except Exception as e:
                                    print(f"Error occurred while creating tracking videos: {str(e)}")

                # 트래킹 작업을 비동기로 시작합니다.
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(create_tracking_videos)

                def check_future():
                    if future.done():
                        set_task_status(task_key, 'completed')
                        future.result()  # 작업 완료를 기다림
                    else:
                        socketio.sleep(1)
                        check_future()
                
                
                socketio.start_background_task(check_future)
                return jsonify({"message": "Tracking video is being created using available images"}), 200

        for person_no in person_nos:
            clip_sql = """
                SELECT c.clip_video, c.clip_time, cam.cam_name
                FROM clip c
                JOIN camera cam ON c.cam_num = cam.cam_num
                WHERE c.person_no = %s
            """
            cursor.execute(clip_sql, (person_no,))
            clip_result = cursor.fetchall()
            clip_info.extend([dict(row) for row in clip_result] if clip_result else [])

    except Exception as e:
        print(f"Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        connection.close()
        if get_task_status(task_key) == 'running':
            set_task_status(task_key, 'failed')
        else:
            set_task_status(task_key, 'completed')
        print(f"Task {task_key} completed or failed")
        return jsonify({"clip_info": clip_info}), 200

# 12. Clip 동영상 스트리밍 엔드포인트 (GET)
@app.route('/stream_clipvideo', methods=['GET'])
def stream_clipvideo():
    video_path = request.args.get('clip_video')
    if not video_path:
        return jsonify({"error": "Invalid clip_video"}), 404

    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404

    reencoded_path = f"{video_path.rsplit('.', 1)[0]}_h264.mp4"
    if not os.path.exists(reencoded_path):
        try:
            reencode_video(video_path, reencoded_path)
        except subprocess.CalledProcessError as e:
            print(f"Error during re-encoding: {str(e)}")
            return jsonify({"error": "Error during re-encoding"}), 500

    def generate():
        with open(reencoded_path, 'rb') as f:
            while True:
                chunk = f.read(1024*1024)
                if not chunk:
                    break
                yield chunk

    range_header = request.headers.get('Range', None)
    if not range_header:
        return Response(generate(), mimetype='video/mp4')

    size = os.path.getsize(reencoded_path)
    byte1, byte2 = 0, None

    if '-' in range_header:
        byte1, byte2 = range_header.replace('bytes=', '').split('-')
    
    byte1 = int(byte1)
    if byte2:
        byte2 = int(byte2)
        length = byte2 - byte1 + 1
    else:
        length = size - byte1

    def generate_range():
        with open(reencoded_path, 'rb') as f:
            f.seek(byte1)
            remaining_length = length
            while remaining_length > 0:
                chunk = f.read(min(1024*1024, remaining_length))
                if not chunk:
                    break
                remaining_length -= len(chunk)
                yield chunk

    rv = Response(generate_range(), 206, mimetype='video/mp4')
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{size}')
    return rv

# 13. 비디오 파일 길이 계산 엔드포인트 (Post)
@app.route('/upload_progresstime', methods=['POST'])
def upload_file_with_progress_time():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "잘못된 요청입니다."}), 400

        user_id = data.get('user_id')
        if not user_id:
            return jsonify({"error": "사용자 ID가 누락되었습니다."}), 400

        # 디렉토리 경로 설정 및 생성
        VIDEO_SAVE_PATH = 'videos'
        IMAGE_SAVE_PATH = 'images'

        user_video_path = os.path.join(VIDEO_SAVE_PATH, str(user_id))
        os.makedirs(user_video_path, exist_ok=True)

        video_data = data.get('video_data', [])
        total_video_length = 0

        # 비디오 파일 처리
        for video in video_data:
            video_name = video.get('video_name')
            video_content = video.get('video_content')

            if not video_name or not video_content:
                return jsonify({"error": "비디오 이름이나 내용이 누락되었습니다."}), 400

            video_path = os.path.join(user_video_path, video_name)

            # 비디오 파일 저장
            with open(video_path, 'wb') as video_file:
                video_file.write(base64.b64decode(video_content))
            
            # 비디오 파일 길이 계산
            video_length = get_video_length(video_path)
            total_video_length += video_length

        # 성공 응답 반환
        return jsonify({"message": "파일이 성공적으로 업로드되었습니다.", "total_length": total_video_length}), 200

    except Exception as e:
        print(f"오류 발생: {e}")
        return jsonify({"error": "파일 업로드 중 오류가 발생했습니다."}), 500
 
# 14. 맵 업데이트 엔드포인트 (Post)
@app.route('/update_cams', methods=['POST'])
def update_cams():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "잘못된 요청입니다."}), 400

        user_id = data.get('userId')
        address = data.get('address')
        markers = data.get('markers')

        print(f"user_id : {user_id}, address : {address}, markers : {markers}")

        user_no = get_user_no(user_id)
        
        if not user_id:
            return jsonify({"error": "사용자 ID가 누락되었습니다."}), 400

        if not markers or not isinstance(markers, list):
            return jsonify({"error": "마커 데이터가 올바르지 않습니다."}), 400

        # 데이터베이스 연결
        connection = get_db_connection()

        with connection.cursor() as cursor:
            # 현재 데이터베이스에 있는 카메라 목록 조회
            sql_select = """
            SELECT cam_name
            FROM camera
            WHERE map_num = (
                SELECT map_num
                FROM map
                WHERE address = %s AND user_no = %s
            )
            """
            cursor.execute(sql_select, (address, user_no))
            db_cam_names = [row['cam_name'] for row in cursor.fetchall()]

            # 마커의 cam_name들 추출
            marker_cam_names = [marker['name'] for marker in markers]

            # UPDATE 또는 INSERT 실행
            for marker in markers:
                cam_name = marker.get('name')
                cam_latitude = marker.get('latitude')
                cam_longitude = marker.get('longitude')

                if cam_name in db_cam_names:
                    # camera 테이블 업데이트
                    sql_update = """
                    UPDATE camera
                    SET cam_latitude = %s, cam_longitude = %s
                    WHERE cam_name = %s
                    AND map_num = (
                        SELECT map_num
                        FROM map
                        WHERE address = %s AND user_no = %s
                    )
                    """
                    cursor.execute(sql_update, (cam_latitude, cam_longitude, cam_name, address, user_no))
                else:
                    # camera 테이블 삽입
                    sql_insert = """
                    INSERT INTO camera (cam_name, cam_latitude, cam_longitude, map_num)
                    VALUES (%s, %s, %s, (
                        SELECT map_num
                        FROM map
                        WHERE address = %s AND user_no = %s
                    ))
                    """
                    cursor.execute(sql_insert, (cam_name, cam_latitude, cam_longitude, address, user_no))

            # DELETE 실행
            for db_cam_name in db_cam_names:
                if db_cam_name not in marker_cam_names:
                    sql_delete = """
                    DELETE FROM camera
                    WHERE cam_name = %s
                    AND map_num = (
                        SELECT map_num
                        FROM map
                        WHERE address = %s AND user_no = %s
                    )
                    """
                    cursor.execute(sql_delete, (db_cam_name, address, user_no))

            # 데이터베이스 변경사항 커밋
            connection.commit()

            # map, ProVideo, bundle 정보 가져오기
            map_camera_sql = """
                SELECT 
                    m.address, 
                    m.map_latitude, 
                    m.map_longitude, 
                    c.cam_name, 
                    c.cam_latitude, 
                    c.cam_longitude, 
                    p.person_id,
                    f.filter_id,
                    f.filter_gender,
                    f.filter_age,
                    f.filter_upclothes,
                    f.filter_downclothes,
                    f.bundle_name
                FROM 
                    map m
                JOIN 
                    camera c ON m.map_num = c.map_num
                LEFT JOIN 
                    origin_video orv ON c.cam_num = orv.cam_num
                LEFT JOIN 
                    person p ON p.or_video_id = orv.or_video_id
                LEFT JOIN 
                    filter f ON f.filter_id = p.filter_id
                WHERE 
                    m.user_no = %s;
            """
            cursor.execute(map_camera_sql, (user_no,))
            map_camera_result = cursor.fetchall()

            map_camera_dict = [dict(row) for row in map_camera_result] if map_camera_result else []

            return jsonify({"map_camera_info": map_camera_dict}), 200

    except Exception as e:
        print(f"오류 발생: {e}")
        return jsonify({"error": "마커 데이터를 처리하는 중 오류가 발생했습니다."}), 500
    
    finally:
        cursor.close()
        connection.close()

# 15. 맵 삭제 엔드포인트 (DELETE)
@app.route('/delete_map', methods=['DELETE'])
def delete_map():
    # user_id와 address를 쿼리 파라미터로 받아옵니다.
    user_id = request.args.get('user_id')
    address = request.args.get('address')
    
    # 필요한 파라미터가 없는 경우 에러를 반환합니다.
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    if not address:
        return jsonify({"error": "Address is required"}), 400

    connection = get_db_connection()
    #cursor = connection.cursor(dictionary=True)

    try:
        with connection.cursor() as cursor:
            # user_id를 기반으로 user_no를 찾습니다.
            user_sql = "SELECT user_no FROM user WHERE user_id = %s"
            cursor.execute(user_sql, (user_id,))
            user_result = cursor.fetchone()

            if user_result is None:
                return jsonify({"error": "User not found"}), 404

            user_no = user_result['user_no']

            # 삭제할 맵 정보를 먼저 가져옵니다.
            select_map_sql = """
                SELECT * FROM map 
                WHERE user_no = %s AND address = %s
            """
            cursor.execute(select_map_sql, (user_no, address))
            delete_map_result = cursor.fetchall()

            if not delete_map_result:
                return jsonify({"error": "Map not found"}), 404

            # 삭제할 맵 정보를 저장합니다.
            delete_map_dict = [dict(row) for row in delete_map_result]

            # 맵 삭제를 수행합니다.
            delete_map_sql = """
                DELETE FROM map 
                WHERE user_no = %s AND address = %s
            """
            cursor.execute(delete_map_sql, (user_no, address))
            connection.commit()

            # 삭제 후 map, camera, person, filter 테이블의 연관된 정보를 가져옵니다.
            map_camera_provideo_sql = """
                SELECT 
                    m.address, 
                    m.map_latitude, 
                    m.map_longitude, 
                    c.cam_name, 
                    c.cam_latitude, 
                    c.cam_longitude, 
                    p.person_id,
                    f.filter_id,
                    f.filter_gender,
                    f.filter_age,
                    f.filter_upclothes,
                    f.filter_downclothes,
                    f.bundle_name
                FROM 
                    map m
                JOIN 
                    camera c ON m.map_num = c.map_num
                LEFT JOIN 
                    origin_video orv ON c.cam_num = orv.cam_num
                LEFT JOIN 
                    person p ON p.or_video_id = orv.or_video_id
                LEFT JOIN 
                    filter f ON f.filter_id = p.filter_id
                WHERE 
                    m.user_no = %s;
            """
            cursor.execute(map_camera_provideo_sql, (user_no,))
            map_camera_provideo_result = cursor.fetchall()
            map_camera_provideo_dict = [dict(row) for row in map_camera_provideo_result]

        # 삭제된 맵 정보와 업데이트된 관련 정보를 반환합니다.
        return jsonify({
            "delete_map_info": delete_map_dict,
            "updated_map_camera_provideo_info": map_camera_provideo_dict
        }), 200

    except Exception as e:
        # 예외 발생 시 오류 메시지를 반환합니다.
        print(f"Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        # 커서와 연결을 닫습니다.
        cursor.close()
        connection.close()

# 16. 비밀번호 업데이트 엔드포인트 (POST)
@app.route('/password_update', methods=['POST'])
def password_update():
    try:
        data = request.get_json()

        # 요청 데이터 유효성 검사
        if not data:
            return jsonify({"error": "잘못된 요청입니다."}), 400
        
        user_data = data.get('user_data', {})
        user_id = user_data.get('user_id')
        user_pw = user_data.get('user_pw')
        new_pw = user_data.get('new_pw')

        # 필수 필드 확인
        if not user_id:
            return jsonify({"error": "사용자 ID가 누락되었습니다."}), 400
        if not user_pw:
            return jsonify({"error": "기존 비밀번호가 누락되었습니다."}), 400
        if not new_pw:
            return jsonify({"error": "새 비밀번호가 누락되었습니다."}), 400

        # DB 연결
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # 유저의 현재 비밀번호 조회
            cursor.execute("SELECT password FROM password LEFT JOIN user ON password.user_no = user.user_no WHERE user_id = %s", (user_id,))
            result = cursor.fetchone()

            if result:
                stored_pw = result['password']  # 저장된 비밀번호

                # 비밀번호 확인 (일반 텍스트 비교)
                if user_pw == stored_pw:
                    # 비밀번호가 일치할 경우, 새로운 비밀번호로 업데이트
                    cursor.execute("UPDATE password SET password = %s WHERE user_no = (SELECT user_no FROM user WHERE user_id = %s)", (new_pw, user_id))
                    connection.commit()

                    return jsonify({"message": "비밀번호가 성공적으로 변경되었습니다."}), 200
                else:
                    return jsonify({"error": "기존 비밀번호가 맞지 않습니다."}), 400
            else:
                return jsonify({"error": "사용자를 찾을 수 없습니다."}), 404

    except Exception as e:
        print(e)
        return jsonify({"error": "서버 오류입니다."}), 500

    finally:
        if 'cursor' in locals():
            cursor.close()


if __name__ == '__main__':
    print("Starting server")  # 서버 시작 디버깅 메시지
    app.run(host="0.0.0.0", port=5002, debug=True)

