import cv2
import os
import pymysql
from datetime import datetime, timedelta
import numpy as np
import sys
from filelock import FileLock

def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='1234',
        database='server',
        cursorclass=pymysql.cursors.DictCursor
    )
#or_video_id를 통해 cam_num을 가져오는 함수를 추가합니다.
def get_cam_num_by_or_video_id(or_video_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT cam_num
                FROM origin_video
                WHERE or_video_id = %s
            """
            cursor.execute(sql, (or_video_id,))
            result = cursor.fetchone()
            if result:
                return result['cam_num']
            else:
                raise ValueError(f"No cam_num found for or_video_id: {or_video_id}")
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

# user_no, video_name 기반 filter_id 조회
def get_or_video_id_by_video_name_and_user_id_and_filter_id(video_name, user_no, filter_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            base_video_name = video_name.split('_person_')[0]
            output_video_name = f"{base_video_name}_output.mp4"
            
            sql = """
                SELECT or_video_id
                FROM processed_video
                WHERE pro_video_name = %s AND user_no = %s AND filter_id = %s
            """
            cursor.execute(sql, (output_video_name, user_no, filter_id))
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

def get_video_start_time(file_label):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT start_time FROM origin_video WHERE or_video_name = %s"
            cursor.execute(sql, (file_label + ".mp4",))
            result = cursor.fetchone()
            if result:
                return result['start_time']
            else:
                raise ValueError(f"No start time found for video: {file_label}.mp4")
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

def get_person_no(person_id, or_video_id, filter_id):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT person_no 
                FROM person 
                WHERE person_id = %s AND or_video_id = %s AND filter_id = %s
            """
            cursor.execute(sql, (person_id, or_video_id, filter_id))
            result = cursor.fetchone()
            if result:
                return result['person_no']
            else:
                raise ValueError(f"No person_no found for person_id: {person_id}, or_video_id: {or_video_id}, and filter_id: {filter_id}")
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
        return None
    finally:
        connection.close()

def process_clips_for_videos(video_names, user_id, user_no, video_person_mapping, filter_id, personid):
    clip_times = []
    video_paths = []

    for video_name in video_names:
        person_no = video_person_mapping.get(video_name)
        if person_no is None:
            print(f"No person_no found for video_name: {video_name}")
            continue

        or_video_ids = get_or_video_id_by_video_name_and_user_id_and_filter_id(video_name, user_no, filter_id)
        if not or_video_ids:
            print(f"No or_video_id found for video_name: {video_name}")
            continue
        
        for or_video_id in or_video_ids:
            video_start_time = get_video_start_time(video_name)
            if video_start_time is None:
                continue

            extracted_dir = os.path.abspath(f'./extracted_images/{user_id}/filter_{filter_id}/{video_name}_clip').replace("\\", "/")
            if not os.path.exists(extracted_dir):
                print(f"Error: Directory {extracted_dir} does not exist.")
                continue

            person_folder_path = os.path.join(extracted_dir, f'person_{personid}').replace("\\", "/")
            if os.path.isdir(person_folder_path):
                video_files = [vf for vf in os.listdir(person_folder_path) if vf.endswith('.mp4')]

                for video_file in video_files:
                    video_file_path = os.path.join(person_folder_path, video_file).replace("\\", "/")
                    video_path_entry = {"file": video_file_path, "start_time": video_start_time, "person_id": personid, "user_no": user_no, "filter_id": filter_id}
                    video_paths.append(video_path_entry)

    process_video_clips(video_paths, user_id, filter_id, clip_times, personid)
    save_clips_to_db(clip_times, video_person_mapping)

def process_video_clips(video_paths, user_id, filter_id, clip_times, personid):
    output_clips_dir = os.path.abspath(f'./extracted_images/{user_id}/filter_{filter_id}/person_{personid}_output_clips').replace("\\", "/")
    os.makedirs(output_clips_dir, exist_ok=True)

    for vp in video_paths:
        if not os.path.exists(vp["file"]):
            print(f"Error: Video file not found at {vp['file']}")
            continue

        cap = cv2.VideoCapture(vp["file"])
        if not cap.isOpened():
            print(f"Error: Could not open video {vp['file']}.")
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        clip_start_frame = None
        clip_index = 0
        combined_writer = None
        file_label = os.path.basename(vp["file"]).split(".")[0]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            lower_blue = np.array([0, 0, 255], dtype=np.uint8)
            upper_blue = np.array([0, 0, 255], dtype=np.uint8)
            mask = cv2.inRange(frame, lower_blue, upper_blue)
            blue_pixel_count = cv2.countNonZero(mask)

            if blue_pixel_count > 0:
                if combined_writer is None:
                    clip_filename = f'{file_label}_{clip_index}.mp4'
                    combined_clip_path = os.path.join(output_clips_dir, clip_filename).replace("\\", "/")
                    combined_writer = cv2.VideoWriter(
                        combined_clip_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        frame_rate,
                        (frame_width, frame_height)
                    )

                if clip_start_frame is None:
                    clip_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                    clip_start_time = vp["start_time"] + timedelta(seconds=(clip_start_frame / frame_rate))
                    clip_times.append({
                        "clip_file": combined_clip_path,
                        "start_time": clip_start_time,
                        "file_label": f"{file_label}_{clip_index}",
                        "video_label": file_label,
                        "person_id": vp["person_id"],
                        "user_no": vp["user_no"],
                        "filter_id": filter_id
                    })

                combined_writer.write(frame)
            else:
                if combined_writer is not None:
                    combined_writer.release()
                    combined_writer = None
                    clip_start_frame = None
                    clip_index += 1

        if combined_writer is not None:
            combined_writer.release()

        cap.release()
        cv2.destroyAllWindows()

    clip_times.sort(key=lambda x: x["start_time"])

    video_order_path = []
    for clip in clip_times:
        if not video_order_path or video_order_path[-1] != clip["video_label"]:
            video_order_path.append(clip["video_label"])

    with open(os.path.join(output_clips_dir, 'clips_times.txt'), 'w') as f:
        for clip_info in clip_times:
            f.write(f"클립 파일: {clip_info['clip_file']}, 시작 시간: {clip_info['start_time']}, person_id: {clip_info['person_id']}, user_no: {clip_info['user_no']}, filter_id: {clip_info['filter_id']}\n")

        f.write("\n비디오 파일 순서:\n")
        video_order_text = ",".join([clip_info['file_label'] for clip_info in clip_times])
        f.write(video_order_text)

        f.write("\n\n이동 경로:\n")
        f.write(",".join(video_order_path))

    print("클립 시간 정보 (정렬된 순서):")
    for clip_info in clip_times:
        print(f"클립 파일: {clip_info['clip_file']}, 시작 시간: {clip_info['start_time']}, person_id: {clip_info['person_id']}, user_no: {clip_info['user_no']}, filter_id: {clip_info['filter_id']}")

    print("클립의 시간 정보가 clips_times.txt 파일로 저장되었습니다.")

def save_clips_to_db(clip_times, video_person_mapping):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            for clip_info in clip_times:
                user_no = clip_info['user_no']
                filter_id = clip_info['filter_id']
                for or_video_id in get_or_video_id_by_video_name_and_user_id_and_filter_id(clip_info['video_label'], user_no, filter_id):
                    person_no = get_person_no(clip_info['person_id'], or_video_id, filter_id)
                    cam_num = get_cam_num_by_or_video_id(or_video_id)

                    if person_no is None:
                        print(f"No person_no found for person_id: {clip_info['person_id']} and or_video_id: {or_video_id}")
                        continue

                    sql_check = """
                        SELECT COUNT(*) as count 
                        FROM clip 
                        WHERE person_no = %s AND clip_video = %s AND clip_time = %s AND cam_num = %s
                    """
                    cursor.execute(sql_check, (person_no, clip_info['clip_file'], clip_info['start_time'], cam_num))
                    count = cursor.fetchone()['count']
                    
                    if count == 0:
                        sql = """
                            INSERT INTO clip (person_no, clip_video, clip_time, cam_num)
                            VALUES (%s, %s, %s, %s)
                        """
                        cursor.execute(sql, (
                            person_no,
                            clip_info['clip_file'],
                            clip_info['start_time'],
                            cam_num
                        ))
                    else:
                        print(f"Clip already exists: {clip_info['clip_file']} at {clip_info['start_time']} for person_no {person_no}")
                    
            connection.commit()
    except pymysql.MySQLError as e:
        print(f"MySQL error occurred: {str(e)}")
    finally:
        connection.close()
        print("클립 정보가 데이터베이스에 저장되었습니다.")

if __name__ == "__main__":
    lock_file_path = '/tmp/clip_processing.lock'
    with FileLock(lock_file_path):
        try:
            user_id = sys.argv[1]
            user_no = sys.argv[2]
            filter_id = sys.argv[3]
            video_names_str = sys.argv[4]
            video_person_mapping_str = sys.argv[5]
            person_id = sys.argv[6]

            video_names = video_names_str.split(',')
            video_person_mapping = {k.strip(): int(v.strip()) for k, v in (item.split(':') for item in video_person_mapping_str.split(','))}

            process_clips_for_videos(video_names, user_id, user_no, video_person_mapping, filter_id, person_id)
        except Exception as e:
            print(f"An error occurred: {str(e)}", file=sys.stderr)
            sys.exit(1)
