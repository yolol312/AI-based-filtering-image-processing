import os
import sys
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

class ImageCleaner:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def extract_embeddings(self, image):
        boxes, probs = self.mtcnn.detect(image)
        if boxes is not None:
            embeddings = []
            for box, prob in zip(boxes, probs):
                if prob < 0.99:
                    continue
                box = [int(coord) for coord in box]
                face = image[box[1]:box[3], box[0]:box[2]]
                if face.size == 0:
                    continue
                if face.shape[0] < 90 or face.shape[1] < 90:
                    continue
                face_tensor = torch.tensor(face).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device) / 255.0
                face_resized = torch.nn.functional.interpolate(face_tensor, size=(160, 160), mode='bilinear')
                embedding = self.resnet(face_resized).detach().cpu().numpy().flatten()
                embeddings.append(embedding)
            return embeddings
        else:
            return []

    def clean_images(self, image_dir):
        for filename in os.listdir(image_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(image_dir, filename)
                image = Image.open(file_path).convert('RGB')
                image_np = np.array(image)
                embeddings = self.extract_embeddings(image_np)
                if not embeddings:
                    os.remove(file_path)
                    print(f"Removed {file_path}")

if __name__ == "__main__":
    video_name = sys.argv[1]
    user_id = sys.argv[2]
    image_dir = f'./extracted_images/{user_id}/{video_name}_face'
    cleaner = ImageCleaner()
    cleaner.clean_images(image_dir)
