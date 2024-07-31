from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os, base64
import pymysql
import subprocess
import threading
import re

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

# 파일 저장 경로 설정
VIDEO_SAVE_PATH = 'uploaded_videos'
IMAGE_SAVE_PATH = 'uploaded_images'

# 디렉토리 생성
os.makedirs(VIDEO_SAVE_PATH, exist_ok=True)
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

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
            person_face_relative_path = os.path.join(user_image_dir, uploaded_image_name)

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
            absolute_video_path = os.path.abspath(video_path)
            sql = "SELECT or_video_id FROM origin_video WHERE or_video_content = %s"
            cursor.execute(sql, (absolute_video_path,))
            result = cursor.fetchone()
            if result:
                return result['or_video_id']
            else:
                print(f"No record found for video_path: {absolute_video_path}")
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
            if person_id_match and gender_match and age_match:
                person_id = person_id_match.group(1)
                gender = gender_match.group(1)
                age = age_match.group(1)
                person_info.append({
                    'person_id': person_id,
                    'gender': gender,
                    'age': age
                })
    return person_info

# 데이터베이스에 데이터 저장 함수
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
                person_image_dir = f'./extracted_images/{user_id}/{or_video_name}_clip/person_{person_id}/'
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
                face_image_relative_path = os.path.join(person_image_dir, face_name)
                sql = """
                    INSERT INTO person (person_id, or_video_id, person_age, person_gender, person_color, person_clothes, person_face, person_origin_face, user_no, filter_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    person_id,
                    or_video_id,
                    person['age'],
                    person['gender'],
                    '',  # person_color
                    '',  # person_clothes
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

# 트래킹 처리 함수
def tracking_video(video_name, user_id, or_video_id, filter_id, saved_paths):
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
                save_processed_video_info2(video_name, user_id, user_no, or_video_id, filter_id)     

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 얼굴 이미지가 주어지지 않았을 때 트래킹 영상 정보 저장
def save_processed_video_info2(video_name, user_id, user_no, or_video_id, filter_id):
    try:
        extracted_dir = f'./extracted_images/{user_id}/{video_name}_clip'
        if not os.path.exists(extracted_dir):
            print(f"No clip folder found for video {video_name}")
            return
        
        pro_video_name = f"{video_name}_output.mp4"
        pro_video_path = os.path.abspath(os.path.join(extracted_dir, pro_video_name))
        
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
    
# 트래킹 영상 DB에 저장 함수 (특정 인물로 판별했을 때 )
def save_processed_video_info(video_name, user_id, user_no, or_video_id, filter_id):
    try:
        extracted_dir = f'./extracted_images/{user_id}/{video_name}_clip'
        if not os.path.exists(extracted_dir):
            print(f"No clip folder found for video {video_name}")
            return
        
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                person_folders = os.listdir(extracted_dir)
                for person_id in person_folders:
                    person_folder_path = os.path.join(extracted_dir, person_id)
                    
                    if os.path.isdir(person_folder_path):
                        video_files = [vf for vf in os.listdir(person_folder_path) if vf.endswith('.mp4')]
                        
                        for video_file in video_files:
                            pro_video_name = f"{video_name}_{person_id}_{video_file}"
                            pro_video_path = os.path.abspath(os.path.join(person_folder_path, video_file))
                            
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

# 얼굴 처리 함수
def process_save_face_info(video_name, user_id, or_video_id, filter_id, clip_flag=True):
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
            info_file_path = f'./extracted_images/{user_id}/{video_name}_face_info.txt'

            # 파싱한 person 정보
            person_info = parse_info_file(info_file_path)

            # DB에 저장
            user_no = get_user_no(user_id)
            if user_no is not None:
                # 이미지 파일 경로 설정
                saved_paths = save_to_db(person_info, or_video_id, user_id, user_no, filter_id)
                print("person DB 저장")
                print("Saved image paths:", saved_paths)
            
            tracking_video(video_name, user_id, or_video_id, filter_id, saved_paths)
            print("pro_video db 저장")
            #if clip_flag:
                #clip_video(video_name, user_id, or_video_id)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# 비디오 처리 함수
def process_video(video_name, user_id, filter_id, clip_flag=True):
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
            video_path = os.path.join('uploaded_videos', user_id, video_name + ".mp4")  # 파일 절대 경로로 변경
            or_video_id = get_or_video_id_by_path(video_path)
            if or_video_id is not None:
                print(f"{video_name} save_face_info")
                process_save_face_info(video_name, user_id, or_video_id, filter_id, clip_flag)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# WebRTC 신호 처리
@socketio.on('offer')
def handle_offer(data):
    print('Offer received:', data)
    emit('offer', data, broadcast=True, include_self=False)

@socketio.on('answer')
def handle_answer(data):
    print('Answer received:', data)
    emit('answer', data, broadcast=True, include_self=False)

@socketio.on('ice-candidate')
def handle_ice_candidate(data):
    print('ICE Candidate received:', data)
    emit('ice-candidate', data, broadcast=True, include_self=False)


# 1. 파일 업로드 엔드포인트(Post)
@app.route('/upload_file', methods=['POST'])
def upload_file():
    try:
        data = request.get_json()
        # JSON 데이터가 제대로 수신되지 않았을 경우 확인
        if not data:
            return jsonify({"status": "error", "message": "No JSON data received"}), 400
        
        # 수신된 데이터가 문자열이 아닌 JSON 객체인지 확인
        if isinstance(data, str):
            return jsonify({"status": "error", "message": "Invalid JSON data format"}), 400

        # 유저 데이터 추출
        user_data = data.get('user_data', {})
        user_id = user_data.get('user_id', '')

        # 필터링 데이터 추출
        filter_data = data.get('filter_data', {})
        age = filter_data.get('age', '')
        gender = filter_data.get('gender', '')
        color = filter_data.get('color', '')
        type = filter_data.get('type', '')
        
        clip_flag = request.form.get('clip_flag', 'true').lower() != 'false'
        #clip_flag = flag_data.get('clip_flag', 'false')
        #face_flag = flag_data.get('face_flag', 'false')

        print(f"Age: {age}")
        print(f"Gender: {gender}")
        print(f"Color: {color}")
        print(f"Type: {type}")
        #print(f"Clip_flag: {clip_flag}")

        # user_id 기반으로 디렉토리 생성
        # 디렉토리 생성
        user_video_path = os.path.join(VIDEO_SAVE_PATH, str(user_id))
        user_image_path = os.path.join(IMAGE_SAVE_PATH, str(user_id))
        os.makedirs(user_video_path, exist_ok=True)
        os.makedirs(user_image_path, exist_ok=True)

        # 데이터베이스 연결
        connection = get_db_connection()
        video_names = []
        filter_id = None
        with connection.cursor() as cursor:
            # 필터 데이터를 DB에 삽입
            filter_sql = """
                INSERT INTO filter (filter_gender, filter_age, filter_color, filter_clothes)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(filter_sql, (gender, age, color, type))
            filter_id = cursor.lastrowid  # 생성된 filter_id 가져오기
            print("filter DB create")
            print(f"filter ID : {filter_id}")
            # 비디오 데이터 처리
            video_data = data.get('video_data', [])
            for video in video_data:
                video_name = video.get('video_name', '')
                video_content_base64 = video.get('video_content', '')
                start_time = video.get('start_time', '')
                cam_name = video.get('cam_name', '')

                if video_name and video_content_base64:
                    video_content = base64.b64decode(video_content_base64)
                    video_path = os.path.join(user_video_path, video_name)
                    absolute_video_path = os.path.abspath(video_path)

                    with open(absolute_video_path, 'wb') as video_file:
                        video_file.write(video_content)

                    # cam_name과 다른 필요한 정보를 이용해 cam_num 조회
                    sql = "SELECT cam_num FROM camera WHERE cam_name = %s AND map_num = (SELECT map_num FROM map WHERE user_no = (SELECT user_no FROM user WHERE user_id = %s))"
                    cursor.execute(sql, (cam_name, user_id))
                    cam_result = cursor.fetchone()
                    if not cam_result:
                        print(f"No cam_num found for cam_name: {cam_name} and user_id: {user_id}")
                        continue
                    
                    cam_num = cam_result['cam_num']

                    # origin_video에 데이터 삽입
                    sql = """
                        INSERT INTO origin_video (or_video_name, or_video_content, start_time, cam_num)
                        VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(sql, (video_name, absolute_video_path, start_time, cam_num))
                    video_names.append(video_name)
                    

            # 이미지 데이터 처리
            image_data = data.get('image_data', {})
            image_name = image_data.get('image_name', '')
            image_content_base64 = image_data.get('image_content', '')

            if image_name and image_content_base64:
                image_content = base64.b64decode(image_content_base64)
                image_path = os.path.join(user_image_path, image_name)
                absolute_image_path = os.path.abspath(image_path)

                with open(absolute_image_path, 'wb') as image_file:
                    image_file.write(image_content)
                print(f"Image: {image_name}")
                print(f"Image: {absolute_image_path}")

            connection.commit()

        connection.close()

        response = jsonify({"status": "success", "message": "Data received and processed successfully"})
        response.status_code = 200

        # process_video 함수에 filter_id를 전달
        for video_name in video_names:
            video_base_name = os.path.splitext(video_name)[0]
            print(f"process_video 실행 : {video_base_name}")
            threading.Thread(target=process_video, args=(video_base_name, user_id, filter_id, clip_flag)).start()

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
                        c.maker_latitude, 
                        c.maker_longitude
                    FROM map m
                    LEFT JOIN camera c ON m.map_num = c.map_num
                    WHERE m.user_no = %s
                """
                cursor.execute(map_camera_sql, (user_no,))
                map_camera_result = cursor.fetchall()

                person_sql = """
                    SELECT 
                        p.person_no,
                        p.person_id,
                        pv.pro_video_id,
                        p.person_age,
                        p.person_gender,
                        p.person_color,
                        p.person_clothes,
                        p.person_face,
                        p.person_origin_face
                    FROM person p
                    JOIN processed_video pv ON p.or_video_id = pv.or_video_id
                    WHERE p.user_no = %s
                """
                cursor.execute(person_sql, (user_no,))
                person_result = cursor.fetchall()

                # Clip 정보 가져오기
                clip_sql = """
                    SELECT 
                        clip_id,
                        person_no,
                        clip_video,
                        clip_location,
                        clip_time
                    FROM clip
                    WHERE person_no IN (SELECT person_no FROM person WHERE user_no = %s)
                """
                cursor.execute(clip_sql, (user_no,))
                clip_result = cursor.fetchall()

                map_camera_dict = [dict(row) for row in map_camera_result]
                person_dict = [dict(row) for row in person_result]
                clip_dict = [dict(row) for row in clip_result]

                response_data = {
                    "message": "Login successful",
                    "user_id": user_id,
                    "user_name": user_name,
                    "map_camera_info": map_camera_dict,
                    "person_info": person_dict,
                    "clip_info": clip_dict  # 클립 정보 추가
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

# 5.지도 마커 위치 엔드포인트(Post)
@app.route('/upload_markers', methods=['POST'])
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

                maker_latitude = camera.get('latitude')
                maker_longitude = camera.get('longitude')

                if cam_name and maker_latitude and maker_longitude:
                    sql = """
                        INSERT INTO camera (cam_name, map_num, maker_latitude, maker_longitude)
                        VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(sql, (cam_name, map_num, maker_latitude, maker_longitude))
                    print(f"Inserted: {cam_name}, {map_num}, {maker_latitude}, {maker_longitude}")
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

if __name__ == '__main__':
    print("Starting server")  # 서버 시작 디버깅 메시지
    app.run(host="0.0.0.0", port=5001, debug=True)