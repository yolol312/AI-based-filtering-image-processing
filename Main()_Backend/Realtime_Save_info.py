import sys
import os
import re
import shutil
import cv2
from collections import Counter

def parse_filename(filename):
    match = re.match(r'person_(\d+)_frame_(\d+)_gender_(\w+)_age_(\w+)_color_(\w+)_clothes_(\w+)\.jpg', filename)
    if match:
        person_id = int(match.group(1))
        frame = int(match.group(2))
        gender = match.group(3).lower()
        age = match.group(4).lower()
        color = match.group(5).lower()
        clothes = match.group(6).lower()
        return person_id, frame, gender, age, color, clothes
    return None

def gather_info_from_files(directory):
    info_dict = {}
    image_files = []

    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            parsed_info = parse_filename(file)
            if parsed_info:
                person_id, frame, gender, age, color, clothes = parsed_info
                
                if person_id not in info_dict:
                    info_dict[person_id] = []
                info_dict[person_id].append((frame, gender, age, color, clothes))
                image_files.append((person_id, os.path.join(directory, file)))

    for person_id in info_dict:
        info_dict[person_id].sort()
    
    return info_dict, image_files

def get_most_common(values):
    counter = Counter(values)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None

def save_info_to_txt(info_dict, output_file, filter_gender, filter_age, filter_color, filter_clothes):
    filtered_persons = []
    with open(output_file, 'w') as f:
        for person_id in sorted(info_dict.keys()):
            frames = [frame for frame, gender, age, color, clothes in info_dict[person_id]]
            genders = [gender for frame, gender, age, color, clothes in info_dict[person_id]]
            ages = [age for frame, gender, age, color, clothes in info_dict[person_id]]
            colors = [color for frame, gender, age, color, clothes in info_dict[person_id]]
            clothes_types = [clothes for frame, gender, age, color, clothes in info_dict[person_id]]
            
            gender_counts = Counter(genders)
            age_counts = Counter(ages)
            color_counts = Counter(colors)
            clothes_counts = Counter(clothes_types)
            
            most_common_gender = get_most_common(genders)
            most_common_age = get_most_common(ages)
            most_common_color = get_most_common(colors)
            most_common_clothes = get_most_common(clothes_types)
            
            print(f"Person {person_id} - Gender Counts: {gender_counts}, Age Counts: {age_counts}")
            print(f"Person {person_id} - Most common gender: {most_common_gender}, Most common age: {most_common_age}")
            print(f"Person {person_id} - Color Counts: {color_counts}, Clothes Counts: {clothes_counts}")
            print(f"Person {person_id} - Most common color: {most_common_color}, Most common clothes: {most_common_clothes}")
            
            if ((filter_gender is None or filter_gender == 'any' or filter_gender == most_common_gender) and 
                (filter_age is None or filter_age == 'any' or filter_age == most_common_age) and 
                (filter_color is None or filter_color == 'any' or filter_color == most_common_color) and 
                (filter_clothes is None or filter_clothes == 'any' or filter_clothes == most_common_clothes)):
                f.write(f'person_{person_id}:\n')
                f.write(f'  gender: {most_common_gender}\n')
                f.write(f'  age: {most_common_age}\n')
                f.write(f'  color: {most_common_color}\n')
                f.write(f'  clothes: {most_common_clothes}\n')
                f.write(f'  frames: {frames}\n')
                f.write(f'  gender_counts: {gender_counts}\n')
                f.write(f'  age_counts: {age_counts}\n')
                f.write(f'  color_counts: {color_counts}\n')
                f.write(f'  clothes_counts: {clothes_counts}\n\n')
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
            _, _, image_gender, image_age, image_color, image_clothes = parsed_info
            
            stored_gender = get_most_common([gender for _, gender, _, _, _ in info_dict[person_id]])
            stored_age = get_most_common([age for _, _, age, _, _ in info_dict[person_id]])
            stored_color = get_most_common([color for _, _, _, color, _ in info_dict[person_id]])
            stored_clothes = get_most_common([clothes for _, _, _, _, clothes in info_dict[person_id]])

            if image_gender != stored_gender or image_age != stored_age or image_color != stored_color or image_clothes != stored_clothes:
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

def create_video(image_paths, output_video_directory, fps=24):
    os.makedirs(output_video_directory, exist_ok=True)
    video_name = "realtime"
    video_output_path = os.path.join(output_video_directory, "realtime.mp4")
    frame_size = (640, 480)
    video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image at {image_path} could not be loaded for video creation.")
            continue
        image_resized = cv2.resize(image, frame_size)
        video_writer.write(image_resized)

    video_writer.release()
    print(f"Video saved to {video_output_path}")

if __name__ == "__main__":
    try:
        video_name = sys.argv[1]
        user_id = sys.argv[2]
        filter_gender = sys.argv[3]
        filter_age = sys.argv[4]
        filter_color = sys.argv[5]
        filter_clothes = sys.argv[6]

        if filter_gender == '여성':
            filter_gender = 'female'
        elif filter_gender == '남성':
            filter_gender = 'male'

        output_image_directory = f"./realtime_extracted_images/{user_id}/"
        output_video_directory = f"./realtime_extracted_videos/{user_id}/"
        image_directory = f"./saved_images/{user_id}/"
        image_paths = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith(('.jpg', '.jpeg', '.png'))]

        os.makedirs(output_image_directory, exist_ok=True)

        for video_folder in os.listdir(output_image_directory):
            if '_face' in video_folder:
                folder_path = os.path.join(output_image_directory, video_folder)
                if os.path.isdir(folder_path):
                    try:
                        create_video(image_paths, output_video_directory)
                        info_dict, image_files = gather_info_from_files(folder_path)

                        output_file = os.path.join(output_image_directory, f"{video_folder}_info.txt")
                        filtered_persons = save_info_to_txt(info_dict, output_file, filter_gender, filter_age, filter_color, filter_clothes)
                        print(f"Information saved to {output_file}")

                        clip_folder = folder_path.replace('_face', '_clip')
                        save_best_faces(image_files, clip_folder, info_dict, filtered_persons)
                    except Exception as e:
                        print(f"Error processing folder {folder_path}: {e}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)