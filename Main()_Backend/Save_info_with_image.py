import sys
import os
import re
import shutil
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
from collections import Counter

# 얼굴 임베딩 모델 초기화
model = InceptionResnetV1(pretrained='vggface2').eval()
transform = transforms.Compose([transforms.Resize((160, 160)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

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

def gather_info_from_files(directory, filter_gender, filter_age, filter_color, filter_clothes):
    info_dict = {}
    image_files = []

    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            parsed_info = parse_filename(file)
            if parsed_info:
                person_id, frame, gender, age, color, clothes = parsed_info

                if (filter_gender is None or filter_gender == 'any' or filter_gender == gender) and \
                   (filter_age is None or filter_age == 'any' or filter_age == age) and \
                   (filter_color is None or filter_color == 'any' or filter_color == color) and \
                   (filter_clothes is None or filter_clothes == 'any' or filter_clothes == clothes):
                    if person_id not in info_dict:
                        info_dict[person_id] = []
                    info_dict[person_id].append((frame, gender, age, color, clothes))
                    image_files.append((person_id, os.path.join(directory, file)))

    for person_id in info_dict:
        info_dict[person_id].sort()  # 프레임을 낮은 순으로 정렬
    
    return info_dict, image_files

def get_most_common(values):
    counter = Counter(values)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None

def save_info_to_txt(info_dict, image_files, target_embedding, output_file, filter_gender, filter_age, filter_color, filter_clothes, threshold=0.6):
    filtered_persons = []
    with open(output_file, 'w') as f:
        for person_id in sorted(info_dict.keys()):
            relevant_files = [file for file in image_files if file[0] == person_id]
            matching_files = []
            for _, filepath in relevant_files:
                embedding = compute_face_embedding(filepath)
                similarity = compare_similarity(embedding, target_embedding)
                if similarity >= threshold:
                    matching_files.append(filepath)

            if matching_files:
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

def compute_face_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(image).numpy().flatten()
    return embedding

def compare_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def save_best_faces(image_files, output_folder, info_dict, filtered_persons, target_embedding, threshold=0.6):
    person_images = {}

    for person_id, filepath in image_files:
        if person_id not in filtered_persons:
            continue

        try:
            embedding = compute_face_embedding(filepath)
            similarity = compare_similarity(embedding, target_embedding)

            if similarity >= threshold:
                image = cv2.imread(filepath)
                quality_score = compute_image_quality(image)

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
        video_name = sys.argv[1]
        user_id = sys.argv[2]
        filter_id = sys.argv[3]
        filter_gender = sys.argv[4].lower()
        filter_age = sys.argv[5].lower()
        filter_color = sys.argv[6].lower()
        filter_clothes = sys.argv[7].lower()
        reference_image_path = sys.argv[8]

        if filter_gender == '여성':
            filter_gender = 'female'
        elif filter_gender == '남성':
            filter_gender = 'male'

        # 'none' 문자열을 None으로 변환
        filter_gender = None if filter_gender == 'none' else filter_gender
        filter_age = None if filter_age == 'none' else filter_age
        filter_color = None if filter_color == 'none' else filter_color
        filter_clothes = None if filter_clothes == 'none' else filter_clothes

        output_directory = f"./extracted_images/{user_id}/filter_{filter_id}/"
        os.makedirs(output_directory, exist_ok=True)

        target_embedding = compute_face_embedding(reference_image_path)

        # 얼굴 정보가 담긴 디렉토리 경로 설정
        face_directory = f"./extracted_images/{user_id}/{video_name}_face"
        if os.path.isdir(face_directory):
            try:
                info_dict, image_files = gather_info_from_files(face_directory, filter_gender, filter_age, filter_color, filter_clothes)

                output_file = os.path.join(output_directory, f"{video_name}_face_info.txt")
                filtered_persons = save_info_to_txt(info_dict, image_files, target_embedding, output_file, filter_gender, filter_age, filter_color, filter_clothes)
                print(f"Information saved to {output_file}")

                clip_folder = os.path.join(output_directory, f"{video_name}_clip")
                save_best_faces(image_files, clip_folder, info_dict, filtered_persons, target_embedding)
            except Exception as e:
                print(f"Error processing folder {face_directory}: {e}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
