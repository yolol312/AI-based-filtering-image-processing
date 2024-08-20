import sys
import os
import re
import shutil
import cv2
from collections import Counter

def parse_filename(filename):
    match = re.match(r'person_(\d+)_frame_(\d+)_gender_(\w+)_age_(\w+)_upclothes_(\w+)_downclothes_(\w+)\.jpg', filename)
    if match:
        person_id = int(match.group(1))
        frame = int(match.group(2))
        gender = match.group(3).lower()
        age = match.group(4).lower()
        upclothes = match.group(5).lower()
        downclothes = match.group(6).lower()
        return person_id, frame, gender, age, upclothes, downclothes
    return None

def gather_info_from_files(directory):
    info_dict = {}
    image_files = []
    
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            parsed_info = parse_filename(file)
            if parsed_info:
                person_id, frame, gender, age, upclothes, downclothes = parsed_info
                if person_id not in info_dict:
                    info_dict[person_id] = []
                info_dict[person_id].append((frame, gender, age, upclothes, downclothes)) ##수정
                image_files.append((person_id, os.path.join(directory, file)))

    for person_id in info_dict:
        info_dict[person_id].sort()  # 프레임을 낮은 순으로 정렬
    
    return info_dict, image_files

def get_most_common(values):
    counter = Counter(values)
    # 가장 많이 나온 값들 중 'unknown'을 제외한 값들 필터링
    most_common_values = [item for item in counter.most_common() if item[0] != 'unknown']
    if most_common_values:
        return most_common_values[0][0]  # 'unknown' 제외한 가장 많이 나온 값 반환
    else:
        return 'unknown'  # 모든 값이 'unknown'일 경우 'unknown' 반환

def save_info_to_txt(info_dict, output_file, filter_gender, filter_age, filter_upclothes, filter_downclothes): 
    filtered_persons = []
    with open(output_file, 'w') as f:
        for person_id in sorted(info_dict.keys()):
            frames = [frame for frame, gender, age, upclothes, downclothes in info_dict[person_id]]            
            genders = [gender for frame, gender, age, upclothes, downclothes in info_dict[person_id]]          
            ages = [age for frame, gender, age, upclothes, downclothes in info_dict[person_id]]               
            upclothes_types = [upclothes for frame, gender, age, upclothes, downclothes in info_dict[person_id]]    
            downclothes_types = [downclothes for frame, gender, age, upclothes, downclothes in info_dict[person_id]] 

            gender_counts = Counter(genders)
            age_counts = Counter(ages)
            upclothes_counts = Counter(upclothes_types)   
            downclothes_counts = Counter(downclothes_types)
            
            most_common_gender = get_most_common(genders)
            most_common_age = get_most_common(ages)
            most_common_upclothes = get_most_common(upclothes_counts)   
            most_common_downclothes = get_most_common(downclothes_counts)   
            
            print(f"Person {person_id} - Gender Counts: {gender_counts}, Age Counts: {age_counts}")
            print(f"Person {person_id} - Most common gender: {most_common_gender}, Most common age: {most_common_age}")
            print(f"Person {person_id} - UpClothes Counts: {upclothes_counts}, DownClothes Counts: {downclothes_counts}")
            print(f"Person {person_id} - Most common upclothes: {most_common_upclothes}, Most common downclothes: {most_common_downclothes}")   
            
            if ((filter_gender is None or filter_gender == 'any' or filter_gender == most_common_gender) and 
                (filter_age is None or filter_age == 'any' or filter_age == most_common_age) and 
                (filter_upclothes is None or filter_upclothes == 'any' or filter_upclothes == most_common_upclothes) and
                (filter_downclothes is None or filter_downclothes == 'any' or filter_downclothes == most_common_downclothes)):
                f.write(f'person_{person_id}:\n')
                f.write(f'  gender: {most_common_gender}\n')
                f.write(f'  age: {most_common_age}\n')
                f.write(f'  upclothes: {most_common_upclothes}\n')
                f.write(f'  downclothes: {most_common_downclothes}\n')
                f.write(f'  frames: {frames}\n')
                f.write(f'  gender_counts: {gender_counts}\n')
                f.write(f'  age_counts: {age_counts}\n')
                f.write(f'  upclothes_counts: {upclothes_counts}\n')
                f.write(f'  downclothes_counts: {downclothes_counts}\n\n')    
                filtered_persons.append(person_id)
                
    return filtered_persons

def compute_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

def save_best_faces(image_files, output_folder, info_dict, filtered_persons):
    person_images = {}

    for person_id, filepath in image_files:
        if person_id not in filtered_persons:
            continue

        try:
            image = cv2.imread(filepath)
            quality_score = compute_image_quality(image)
            
            parsed_info = parse_filename(os.path.basename(filepath))
            if not parsed_info:
                continue
            _, _, image_gender, image_age, image_upclothes, image_downclothes = parsed_info
            
            stored_gender = get_most_common([gender for _, gender, _, _, _ in info_dict[person_id]])
            stored_age = get_most_common([age for _, _, age, _, _ in info_dict[person_id]])
            stored_upclothes = get_most_common([upclothes for _, _,  _, upclothes, _ in info_dict[person_id]])
            stored_downclothes = get_most_common([downclothes for _, _, _, _, downclothes in info_dict[person_id]])

            if image_gender != stored_gender or image_age != stored_age or image_upclothes != stored_upclothes or image_downclothes != stored_downclothes:  
                continue

            if person_id not in person_images:
                person_images[person_id] = (filepath, quality_score)
            else:
                if quality_score > person_images[person_id][1]:
                    person_images[person_id] = (filepath, quality_score)
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    for person_id, (best_image_path, _) in person_images.items():
        person_folder = os.path.join(output_folder, f"person_{person_id}")
        os.makedirs(person_folder, exist_ok=True)
        output_filepath = os.path.join(person_folder, os.path.basename(best_image_path))
        shutil.copy(best_image_path, output_filepath)
        print(f"Copied best quality face image: {output_filepath}")

if __name__ == "__main__":
    try:
        video_name = "W17-3"
        user_id = "modelTest"
        filter_id = "1"
        filter_gender = "none"
        filter_age = "none"
        filter_upclothes = "none"
        filter_downclothes = "none"


        #video_name = sys.argv[1]
        #user_id = sys.argv[2]
        #filter_id = sys.argv[3]
        #filter_gender = sys.argv[4].lower()
        #filter_age = sys.argv[5].lower()
        #filter_upclothes = sys.argv[6].lower()
        #filter_downclothes = sys.argv[7].lower()

        if filter_gender == '여성':
            filter_gender = 'female'
        elif filter_gender == '남성':
            filter_gender = 'male'

        # 'none' 문자열을 None으로 변환
        filter_gender = None if filter_gender == 'none' else filter_gender
        filter_age = None if filter_age == 'none' else filter_age
        filter_upclothes = None if filter_upclothes == 'none' else filter_upclothes
        filter_downclothes = None if filter_downclothes == 'none' else filter_downclothes
        

        output_directory = f"./extracted_images/{user_id}/filter_{filter_id}/"
        os.makedirs(output_directory, exist_ok=True)  # 필터 디렉토리를 미리 생성합니다.
        
        user_directory = f"./extracted_images/{user_id}/"
        for video_folder in os.listdir(user_directory):
            if '_face' in video_folder and video_name in video_folder:  # video_name과 일치하는 폴더만 처리
                folder_path = os.path.join(user_directory, video_folder)
                if os.path.isdir(folder_path):
                    try:
                        info_dict, image_files = gather_info_from_files(folder_path) # 필터링 제거하고 딕셔너리와 이미지 파일만 생성

                        output_file = os.path.join(output_directory, f"{video_folder}_info.txt")
                        filtered_persons = save_info_to_txt(info_dict, output_file, filter_gender, filter_age, filter_upclothes, filter_downclothes) # 필터링을 해당 메서드에 넣음
                        print(f"Information saved to {output_file}")

                        clip_folder = os.path.join(output_directory, video_folder.replace('_face', '_clip'))
                        os.makedirs(clip_folder, exist_ok=True)  # 클립 폴더를 생성
                        save_best_faces(image_files, clip_folder, info_dict, filtered_persons)
                    except Exception as e:
                        print(f"Error processing folder {folder_path}: {e}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
