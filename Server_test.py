from flask import Flask, request, jsonify, Response
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app)

# 데이터베이스 연결 함수
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='1234',
        database='wb39_project2',
        cursorclass=pymysql.cursors.DictCursor
    )

# 각 웹캠의 이미지가 저장될 폴더 경로 설정
SAVE_FOLDER = os.path.abspath('saved_images').replace("\\", "/")
WEBCAM_FOLDERS = [os.path.abspath(f"webcam_{i}").replace("\\", "/") for i in range(4)]

# 폴더가 없으면 생성
for folder in WEBCAM_FOLDERS:
    os.makedirs(folder, exist_ok=True)

# 파일 저장 경로 설정
VIDEO_SAVE_PATH = os.path.abspath('uploaded_videos').replace("\\", "/")
IMAGE_SAVE_PATH = os.path.abspath('uploaded_images').replace("\\", "/")

# 디렉토리 생성
os.makedirs(VIDEO_SAVE_PATH, exist_ok=True)
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

#person_no 에 해당하는 필터 정보 가져오기
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

#클립추출을 위한 트래킹 영상이 존재하는지 확인
def does_video_file_exist(user_id, video_name, person_id):
    video_dir = os.path.abspath(f'./extracted_images/{user_id}/{video_name}_clip/person_{person_id}/').replace("\\", "/")
    if not os.path.exists(video_dir):
        print(f"Directory does not exist: {video_dir}")
        return False

    # 디렉토리 내의 비디오 파일 찾기 (.mp4 확장자)
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
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

# pro_video_content를 DB에서 얻어오는 함수
def get_video_path(user_id, pro_video_name):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # 사용자 번호 가져오기
            cursor.execute("SELECT user_no FROM user WHERE user_id = %s", (user_id,))
            user = cursor.fetchone()
            if not user:
                return None
            user_no = user['user_no']
            
            # 동영상 경로 가져오기
            cursor.execute("SELECT pro_video_content FROM processed_video WHERE user_no = %s AND pro_video_name = %s", (user_no, pro_video_name))
            video = cursor.fetchone()
            if not video:
                return None
            return video['pro_video_content']
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
            user_image_dir = os.path.abspath(f'./uploaded_images/{user_id}/').replace("\\", "/")
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
                return result['or_video_name']
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
            color_match = re.search(r'color: (\w+)', person)
            clothes_match = re.search(r'clothes: (\w+)', person)
            if person_id_match and gender_match and age_match:
                person_id = person_id_match.group(1)
                gender = gender_match.group(1)
                age = age_match.group(1)
                color = color_match.group(1)
                clothes = clothes_match.group(1)
                person_info.append({
                    'person_id': person_id,
                    'gender': gender,
                    'age': age,
                    'color': color,
                    'clothes': clothes
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
                
                # .mp4 확장자 제거
                or_video_name = os.path.splitext(or_video_name_result)[0]  # or_video_name_result이 문자열일 경우

                person_id = person['person_id']
                # 이미지 파일 경로 설정
                person_image_dir = os.path.abspath(f'./extracted_images/{user_id}/{or_video_name}_clip/person_{person_id}/').replace("\\", "/")
                if not os.path.exists(person_image_dir):
                    print(f"Directory not found: {person_image_dir}")
                    continue

                # 디렉토리 내의 이미지 파일 찾기
                face_files = [f for f in os.listdir(person_image_dir) if f.endswith('.jpg') or f.endswith('.png')]
                if not face_files:
                    print(f"No image files found in directory: {person_image_dir}")
                    continue

                # 첫 번째 이미지를 사용 (필요에 따라 다른 선택 방법 사용 가능)
                face_name = face_files[0]

                # 상대 경로로 저장
                face_image_relative_path = os.path.join(person_image_dir, face_name).replace("\\", "/")
                sql = """
                    INSERT INTO person (person_id, or_video_id, person_age, person_gender, person_color, person_clothes, person_face, person_origin_face, user_no, filter_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    person_id,
                    or_video_id,
                    person['age'],
                    person['gender'],
                    person['color'],  # person_color
                    person['clothes'],  # person_clothes
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
    try:
        with connection.cursor() as cursor:
            for person in person_info:
                or_video_name_result = get_or_video_name(or_video_id)
                if not or_video_name_result:
                    print(f"No or_video_name found for or_video_id: {or_video_id}")
                    continue
                
                # .mp4 확장자 제거
                or_video_name = os.path.splitext(or_video_name_result)[0]  # or_video_name_result이 문자열일 경우

                person_id = person['person_id']
                # 이미지 파일 경로 설정
                person_image_dir = os.path.abspath(f'./extracted_images/{user_id}/{or_video_name}_clip/person_{person_id}/').replace("\\", "/")
                if not os.path.exists(person_image_dir):
                    print(f"Directory not found: {person_image_dir}")
                    continue

                # 디렉토리 내의 이미지 파일 찾기
                face_files = [f for f in os.listdir(person_image_dir) if f.endswith('.jpg') or f.endswith('.png')]
                if not face_files:
                    print(f"No image files found in directory: {person_image_dir}")
                    continue

                # 첫 번째 이미지를 사용 (필요에 따라 다른 선택 방법 사용 가능)
                face_name = face_files[0]

                # 상대 경로로 저장
                face_image_relative_path = os.path.join(person_image_dir, face_name).replace("\\", "/")
                sql = """
                    INSERT INTO person (person_id, or_video_id, person_age, person_gender, person_color, person_clothes, person_face, person_origin_face, user_no, filter_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                # image_path 앞에 './'를 추가하고, 역슬래시를 슬래시로 변경
                image_path = os.path.abspath(image_path).replace("\\", "/")
                cursor.execute(sql, (
                    person_id,
                    or_video_id,
                    person['age'],
                    person['gender'],
                    person['color'],  # person_color
                    person['clothes'],  # person_clothes
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
def clip_video(video_name, user_id, or_video_id):
    try:
        user_no = get_user_no(user_id)
        if user_no is not None:
            process = subprocess.Popen(
                ["python", "Clip.py", video_name, str(user_id), str(user_no), str(or_video_id)], 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
                
            if process.returncode != 0:
                print(f"Error occurred: {stderr.decode('utf-8')}")
            else:
                print("클립 추출 성공")
                # 각 클립에 대해 update_person_face_from_clip 호출
                connection = get_db_connection()
                with connection.cursor() as cursor:
                    sql = "SELECT person_no FROM clip WHERE person_no IN (SELECT person_no FROM person WHERE or_video_id = %s)"
                    cursor.execute(sql, (or_video_id,))
                    person_nos = cursor.fetchall()
                    for person_no in person_nos:
                        update_person_face_from_clip(person_no['person_no'])

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 트래킹 처리 함수 (이미지 없을 때)
def tracking_video_without_image(video_name, user_id, or_video_id, filter_id, saved_paths):
    try:
        # Join the paths into a single string, separating each path with a comma or another delimitery
        paths_str = ','.join(saved_paths)
        print(f"{video_name} => Tracking  - - - paths : {paths_str}")
        process = subprocess.Popen(
            ["python", "Tracking.py", video_name, str(user_id), paths_str], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error occurred: {stderr.decode('utf-8')}")
        else:
            print(f"{video_name} 트래킹 영상 추출 성공")
            user_no = get_user_no(user_id)
            if user_no is not None:
                save_processed_video_info_without_image(video_name, user_id, user_no, or_video_id, filter_id)     

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 트래킹 처리 함수 (이미지 있을 때)
def tracking_video_with_image(video_name, user_id, or_video_id, filter_id, saved_paths):
    try:
        # Join the paths into a single string, separating each path with a comma or another delimitery
        paths_str = ','.join(saved_paths)
        print(f"{video_name} => Tracking  - - - paths : {paths_str}")
        process = subprocess.Popen(
            ["python", "Tracking_with_image.py", video_name, str(user_id), paths_str], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error occurred: {stderr.decode('utf-8')}")
        else:
            print(f"{video_name} 트래킹 영상 추출 성공")
            user_no = get_user_no(user_id)
            if user_no is not None:
                save_processed_video_info_with_image(video_name, user_id, user_no, or_video_id, filter_id)     

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 트래킹 영상 정보 저장 (이미지 없을 때)
def save_processed_video_info_without_image(video_name, user_id, user_no, or_video_id, filter_id):
    try:
        extracted_dir = os.path.abspath(f'./extracted_images/{user_id}/{video_name}_clip').replace("\\", "/")
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
                    WHERE pro_video_name = %s AND or_video_id = %s
                """
                cursor.execute(sql_check, (pro_video_name, or_video_id))
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
    
# 트래킹 영상 정보 저장 (이미지 있을 때)
def save_processed_video_info_with_image(video_name, user_id, user_no, or_video_id, filter_id):
    try:
        extracted_dir = os.path.abspath(f'./extracted_images/{user_id}/{video_name}_clip').replace("\\", "/")
        if not os.path.exists(extracted_dir):
            print(f"No clip folder found for video {video_name}")
            return
        
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                person_folders = os.listdir(extracted_dir)
                for person_id in person_folders:
                    person_folder_path = os.path.abspath(os.path.join(extracted_dir, person_id)).replace("\\", "/")
                    
                    if os.path.isdir(person_folder_path):
                        video_files = [vf for vf in os.listdir(person_folder_path) if vf.endswith('.mp4')]
                        
                        for video_file in video_files:
                            pro_video_name = f"{video_name}_{video_file}"
                            pro_video_path = os.path.abspath(os.path.join(person_folder_path, video_file)).replace("\\", "/")
                            # 중복 체크 로직 추가
                            sql_check = """
                                SELECT COUNT(*) as count FROM processed_video 
                                WHERE pro_video_name = %s AND or_video_id = %s
                            """
                            cursor.execute(sql_check, (pro_video_name, or_video_id))
                            count = cursor.fetchone()['count']
                            
                            if count == 0:
                                sql = """
                                    INSERT INTO processed_video (or_video_id, pro_video_name, pro_video_content, user_no, filter_id)
                                    VALUES (%s, %s, %s, %s, %s)
                                """
                                cursor.execute(sql, (or_video_id, pro_video_name, pro_video_path, user_no, filter_id))
                connection.commit()
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
            filter_color = filter_info['filter_color']
            filter_clothes = filter_info['filter_clothes']
        else:
            print(f"No filter found for filter_id: {filter_id}")
            return
        # save_face_info6.py 스크립트 호출 (백그라운드 실행)
        process = subprocess.Popen(
            ["python", "Save_info.py", video_name, str(user_id), filter_gender, filter_age, filter_color, filter_clothes], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error occurred: {stderr.decode('utf-8')}")
        else:
            print(f"{video_name} Save_info.py 정보 추출 성공")
            # 예시 메모장 파일 경로
            info_file_path = os.path.abspath(f'./extracted_images/{user_id}/{video_name}_face_info.txt').replace("\\", "/")

            # 파싱한 person 정보
            person_info = parse_info_file(info_file_path)

            # DB에 저장
            user_no = get_user_no(user_id)
            if user_no is not None:
                # 이미지 파일 경로 설정
                saved_paths = save_to_db(person_info, or_video_id, user_id, user_no, filter_id)
                print("person DB 저장")
                print("Saved image paths:", saved_paths)
            
            tracking_video_without_image(video_name, user_id, or_video_id, filter_id, saved_paths)
            print("pro_video db 저장")
            #if clip_flag:
                #clip_video(video_name, user_id, or_video_id)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 얼굴 처리 함수 (이미지 있을 때)
def process_save_face_info_with_image(video_name, user_id, or_video_id, filter_id, image_path, clip_flag=True):
    try:
        # filter 정보 가져오기
        filter_info = get_filter_info(filter_id)
        if filter_info:
            filter_gender = filter_info['filter_gender']
            filter_age = filter_info['filter_age']
            filter_color = filter_info['filter_color']
            filter_clothes = filter_info['filter_clothes']
        else:
            print(f"No filter found for filter_id: {filter_id}")
            return
        # save_face_info6.py 스크립트 호출 (백그라운드 실행)
        process = subprocess.Popen(
            ["python", "Save_info_with_image.py", video_name, str(user_id), filter_gender, filter_age, filter_color, filter_clothes, image_path], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error occurred: {stderr.decode('utf-8')}")
        else:
            print(f"{video_name} Save_info.py 정보 추출 성공")
            # 예시 메모장 파일 경로
            info_file_path = os.path.abspath(f'./extracted_images/{user_id}/{video_name}_face_info.txt').replace("\\", "/")

            # 파싱한 person 정보
            person_info = parse_info_file(info_file_path)

            # DB에 저장
            user_no = get_user_no(user_id)
            if user_no is not None:
                # 이미지 파일 경로 설정
                saved_paths = save_to_db_with_image(person_info, or_video_id, user_id, user_no, filter_id, image_path)
                print("person DB 저장")
                print("Saved image paths:", saved_paths)
            
            tracking_video_with_image(video_name, user_id, or_video_id, filter_id, saved_paths)
            print("pro_video db 저장")
            #if clip_flag:
                #clip_video(video_name, user_id, or_video_id)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 비디오 처리 함수 (이미지 없을 때)
def process_video_without_images(video_name, user_id, filter_id, clip_flag=True):
    try:
        # Main_image2.py 스크립트 호출 (백그라운드 실행)
        process = subprocess.Popen(
            ["python", "Main.py", video_name, str(user_id)], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error occurred: {stderr.decode('utf-8')}")
        else:
            print("Main.py 얼굴정보추출 성공")
            # 얼굴정보추출 성공 후 save_face_info6.py 실행
            video_path = os.path.abspath(os.path.join('uploaded_videos', user_id, video_name + ".mp4")).replace("\\", "/")
            or_video_id = get_or_video_id_by_path(video_path)
            if or_video_id is not None:
                process_save_face_info_without_image(video_name, user_id, or_video_id, filter_id, clip_flag)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 비디오 처리 함수 (이미지 있을 때)
def process_video_with_images(video_name, user_id, filter_id, image_path, clip_flag=True):
    try:
        # Main_image2.py 스크립트 호출 (백그라운드 실행)
        process = subprocess.Popen(
            ["python", "Main.py", video_name, str(user_id)], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error occurred: {stderr.decode('utf-8')}")
        else:
            print("Main.py 얼굴정보추출 성공")
            # 얼굴정보추출 성공 후 save_face_info6.py 실행
            video_path = os.path.abspath(os.path.join('uploaded_videos', user_id, video_name + ".mp4")).replace("\\", "/")
            or_video_id = get_or_video_id_by_path(video_path)
            if or_video_id is not None:
                process_save_face_info_with_image(video_name, user_id, or_video_id, filter_id, image_path, clip_flag)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 0. 실시간 웹캠 이미지 전송(Post)
@app.route('/upload_image_<int:webcam_id>', methods=['POST'])
def upload_image(webcam_id):

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

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Image decoding failed"}), 500
    

    # 이미지 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    videoname = f"{user_id}_realtime"
    filename = f"{timestamp}_{videoname}.jpg"
    folder_path = os.path.abspath(os.path.join(SAVE_FOLDER, WEBCAM_FOLDERS[webcam_id])).replace("\\", "/")
    filepath = os.path.abspath(os.path.join(folder_path, filename)).replace("\\", "/")
    cv2.imwrite(filepath, img)
    
    print(f"Received and saved image from webcam {webcam_id} with shape: {img.shape} as {filename}")
    
    return jsonify({"message": "Image received and saved"}), 200

# clip_video 출력을 위한 Person정보 입력받기
@app.route('/get_Person_to_clip', methods=['GET'])
def get_Person_to_clip():
    user_id = request.args.get('user_id')
    person_id = request.args.get('person_id')  # person_id를 추가로 받아야 합니다.

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    if not person_id:
        return jsonify({"error": "Person ID is required"}), 400

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # user_id를 기반으로 user_no를 찾기
        user_sql = "SELECT user_no FROM user WHERE user_id = %s"
        cursor.execute(user_sql, (user_id,))
        user_result = cursor.fetchone()

        if user_result is None:
            return jsonify({"error": "User not found"}), 404

        user_no = user_result['user_no'] if isinstance(user_result, dict) else user_result[0]

        # person_id를 기반으로 person_no를 찾기
        person_sql = "SELECT person_no FROM person WHERE person_id = %s"
        cursor.execute(person_sql, (person_id,))
        person_result = cursor.fetchone()

        if person_result is None:
            return jsonify({"error": "Person not found"}), 404

        person_no = person_result['person_no'] if isinstance(person_result, dict) else person_result[0]

        #person_no 로 or_video_id를 찾기

        or_video_id = get_or_video_id_by_person_no(person_no)
        #or_video_id로 or_video_name 찾기
        video_name = get_or_video_name(or_video_id)

        # 경로에 비디오 파일이 존재하는지 확인
        if not does_video_file_exist(user_id, video_name, person_id):
            # 비디오 파일이 없을 경우, 해당 디렉토리에서 이미지 파일을 찾아 트래킹 비디오 생성
            image_dir = os.path.abspath(f'./extracted_images/{user_id}/{video_name}_clip/person_{person_id}/').replace("\\", "/")
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
            if not image_files:
                return jsonify({"error": "No image files found to create tracking video"}), 404

            filter_id = get_filter_id_by_person_no(person_no)
            # 첫 번째 이미지를 사용하여 트래킹 비디오 생성 (필요에 따라 다른 선택 방법 사용 가능)
            image_path = os.path.abspath(os.path.join(image_dir, image_files[0])).replace("\\", "/")
            tracking_video_with_image(video_name, user_id, or_video_id, filter_id, image_path)
            return jsonify({"message": "Tracking video created using available images"}), 200
        
        #클립 생성 메서드 (해당 person_id 의 디렉토리에 그 person_id 가 트래킹 된 영상이 존재해야함)
        clip_video(video_name, user_id, or_video_id)
        
        # person_no를 기반으로 클립 정보 가져오기
        clip_sql = """
            SELECT clip_video, clip_time FROM clip WHERE person_no = %s
            """
        cursor.execute(clip_sql, (person_no,))
        clip_result = cursor.fetchall()

        clip_info = [dict(row) for row in clip_result] if clip_result else []

        return jsonify({"clip_info": clip_info}), 200

    except Exception as e:
        print(f"Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()

# Pro_video 동영상 스트리밍 엔드포인트
@app.route('/stream_video', methods=['GET'])
def stream_video():
    user_id = request.args.get('user_id')
    pro_video_name = request.args.get('pro_video_name')
    print("user_id:", user_id)  # 디버깅 메시지 추가
    print("pro_video_name:", pro_video_name)  # 디버깅 메시지 추가

    video_path = get_video_path(user_id, pro_video_name)
    if not video_path:
        return jsonify({"error": "Invalid user_id or pro_video_name"}), 404

    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404

    def generate():
        with open(video_path, 'rb') as f:
            while True:
                chunk = f.read(1024*1024)
                if not chunk:
                    break
                yield chunk

    range_header = request.headers.get('Range', None)
    if not range_header:
        return Response(generate(), mimetype='video/mp4')

    size = os.path.getsize(video_path)
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
        with open(video_path, 'rb') as f:
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

# clip_video 동영상 스트리밍 엔드포인트
@app.route('/stream_clipvideo', methods=['GET'])
def stream_clipvideo():
    video_path = request.args.get('clip_video')
    if not video_path:
        return jsonify({"error": "Invalid clip_video"}), 404

    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404

    def generate():
        with open(video_path, 'rb') as f:
            while True:
                chunk = f.read(1024*1024)
                if not chunk:
                    break
                yield chunk

    range_header = request.headers.get('Range', None)
    if not range_header:
        return Response(generate(), mimetype='video/mp4')

    size = os.path.getsize(video_path)
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
        with open(video_path, 'rb') as f:
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

# 1. 파일 업로드 엔드포인트(Post)
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

        filter_data = data.get('filter_data', {})
        age = filter_data.get('age', '')
        gender = filter_data.get('gender', '')
        color = filter_data.get('color', '')
        type = filter_data.get('type', '')

        clip_flag = request.form.get('clip_flag', 'true').lower() != 'false'

        user_video_path = os.path.abspath(os.path.join(VIDEO_SAVE_PATH, str(user_id))).replace("\\", "/")
        user_image_path = os.path.abspath(os.path.join(IMAGE_SAVE_PATH, str(user_id))).replace("\\", "/")
        os.makedirs(user_video_path, exist_ok=True)
        os.makedirs(user_image_path, exist_ok=True)

        connection = get_db_connection()
        video_names = []
        filter_id = None
        with connection.cursor() as cursor:
            filter_sql = """
                INSERT INTO filter (filter_gender, filter_age, filter_color, filter_clothes)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(filter_sql, (gender, age, color, type))
            filter_id = cursor.lastrowid
            print("filter DB create")
            print(f"filter ID : {filter_id}")

            video_data = data.get('video_data', [])
            for video in video_data:
                video_name = video.get('video_name', '')
                video_content_base64 = video.get('video_content', '')
                start_time = video.get('start_time', '')
                cam_name = video.get('cam_name', '')

                if video_name and video_content_base64:
                    video_content = base64.b64decode(video_content_base64)
                    video_path = os.path.abspath(os.path.join(user_video_path, video_name)).replace("\\", "/")
                    with open(video_path, 'wb') as video_file:
                        video_file.write(video_content)

                    # cam_name과 다른 필요한 정보를 이용해 cam_num 조회
                    sql = """
                        SELECT cam_num 
                        FROM camera 
                        WHERE cam_name = %s 
                        AND map_num IN (SELECT map_num FROM map WHERE user_no = (SELECT user_no FROM user WHERE user_id = %s))
                    """
                    cursor.execute(sql, (cam_name, user_id))
                    cam_results = cursor.fetchall()
                    if not cam_results:
                        print(f"No cam_num found for cam_name: {cam_name} and user_id: {user_id}")
                        continue
                    
                    for cam_result in cam_results:
                        cam_num = cam_result['cam_num']

                        # origin_video에 데이터 삽입
                        sql = """
                            INSERT INTO origin_video (or_video_name, or_video_content, start_time, cam_num)
                            VALUES (%s, %s, %s, %s)
                        """
                        cursor.execute(sql, (video_name, video_path, start_time, cam_num))
                        video_names.append(video_name)
            
            image_data = data.get('image_data', {})
            image_name = image_data.get('image_name', '')
            image_content_base64 = image_data.get('image_content', '')

            image_path = None
            if image_name and image_content_base64:
                image_content = base64.b64decode(image_content_base64)
                image_path = os.path.abspath(os.path.join(user_image_path, image_name)).replace("\\", "/")

                with open(image_path, 'wb') as image_file:
                    image_file.write(image_content)
                print(f"Image: {image_name}")
                print(f"Image: {image_path}")

            connection.commit()
            connection.close()

            response = jsonify({"status": "success", "message": "Data received and processed successfully"})
            response.status_code = 200

            for video_name in video_names:
                video_base_name = os.path.splitext(video_name)[0]
                if image_path:
                    threading.Thread(target=process_video_with_images, args=(video_base_name, user_id, filter_id, image_path, clip_flag)).start()
                else:
                    threading.Thread(target=process_video_without_images, args=(video_base_name, user_id, filter_id, clip_flag)).start()

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

# 2.회원가입 엔드포인트(Post)
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
    
# 3. 로그인 엔드포인트(Post)
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    print("Login attempt:", data)
    if data and 'ID' in data and 'PW' in data:
        connection = get_db_connection()
        cursor = connection.cursor()

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
                    SELECT user_name
                    FROM profile
                    WHERE user_no = %s
                """
                cursor.execute(profile_sql, (user_no,))
                profile_result = cursor.fetchone()
                user_name = profile_result['user_name'] if profile_result else "Unknown"

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

                # ProVideo, Person 정보 가져오기
                provideo_person_sql = """
                    SELECT 
                        pro_video_name, 
                        video.filter_id, 
                        person_id, 
                        person_age, 
                        person_gender, 
                        person_color, 
                        person_clothes, 
                        person_face
                    FROM user, person, processed_video AS video
                    WHERE user.user_no = person.user_no 
                        AND user.user_no = video.user_no 
                        AND person.or_video_id = video.or_video_id 
                        AND person.filter_id = video.filter_id
                        AND user.user_no = %s
                """
                cursor.execute(provideo_person_sql, (user_no,))
                provideo_person_result = cursor.fetchall()

                # Cam Name 정보 가져오기
                camname_sql = """
                    SELECT 
                        cam_name
                    FROM user, processed_video as p_video, origin_video as o_video, camera
                    WHERE user.user_no = p_video.user_no
                    AND p_video.or_video_id = o_video.or_video_id
                    AND o_video.cam_num = camera.cam_num
                    AND user.user_no = %s
                """
                cursor.execute(camname_sql, (user_no,))
                camname_result = cursor.fetchall()

                map_camera_dict = [dict(row) for row in map_camera_result] if map_camera_result else []
                provideo_person_dict = [dict(row) for row in provideo_person_result] if provideo_person_result else []
                camname_dict = [dict(row) for row in camname_result] if camname_result else []

                response_data = {
                    "message": "Login successful",
                    "user_id": user_id,
                    "user_name": user_name,
                    "map_camera_info": map_camera_dict,
                    "provideo_person_info": provideo_person_dict,
                    "camname_info": camname_dict,
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
    
# 4.지도 주소 엔드포인트(Post)
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
                cursor.execute("SELECT user_no FROM user WHERE user_id = %s", (user_id,))
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

# 5.지도 마커 위치 엔드포인트(Post)
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

# 6.지도 업데이트 엔드포인트 (GET)
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

# 7.ProVideo_Person 업데이트 엔드포인트 (GET)
@app.route('/update_pro_person', methods=['GET'])
def upload_pro_person():
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

        # user_no를 기반으로 ProVideo와 Person을 가져오기
        # ProVideo, Person 정보 가져오기
        provideo_person_sql = """
            SELECT 
                pro_video_name, 
                video.filter_id, 
                person_id, 
                person_age, 
                person_gender, 
                person_color, 
                person_clothes, 
                person_face
            FROM user, person, processed_video AS video
            WHERE user.user_no = person.user_no 
                AND user.user_no = video.user_no 
                AND person.or_video_id = video.or_video_id 
                AND person.filter_id = video.filter_id
                AND user.user_no = %s
            """
        cursor.execute(provideo_person_sql, (user_no,))
        provideo_person_result = cursor.fetchall()

        provideo_person_dict = [dict(row) for row in provideo_person_result] if provideo_person_result else []

        return jsonify({"provideo_person_dict": provideo_person_dict}), 200

    except Exception as e:
        print(f"Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()

if __name__ == '__main__':
    print("Starting server")  # 서버 시작 디버깅 메시지
    app.run(host="0.0.0.0", port=5001, debug=True)
