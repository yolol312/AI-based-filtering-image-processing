import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from pycocotools.coco import COCO
import random
import time
from ultralytics import YOLO
import json
from torchvision.ops import nms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Activation Function
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Convolutional Layer
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Bottleneck Layer
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        hidden_channels = out_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.cv2 = Conv(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))

# C2f Layer
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks):
        super(C2f, self).__init__()
        hidden_channels = out_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.cv2 = Conv(hidden_channels * (num_bottlenecks + 1), out_channels, kernel_size=1, stride=1, padding=0)
        self.bottlenecks = nn.ModuleList([Bottleneck(hidden_channels, hidden_channels) for _ in range(num_bottlenecks)])

    def forward(self, x):
        y = [self.cv1(x)]
        for bottleneck in self.bottlenecks:
            y.append(bottleneck(y[-1]))
        return self.cv2(torch.cat(y, 1))

# SPPF Layer
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.cv2 = Conv(hidden_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

# Detect Layer for Single Class and Single Scale Prediction
class Detect(nn.Module):
    def __init__(self, num_classes=1):
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            Conv(1280, 640, 3, 1, 1),
            Conv(640, 320, 3, 1, 1),
            nn.Conv2d(320, num_classes + 5, 1, 1)
        )

    def forward(self, x):
        return self.conv(x)

# YOLOv8x Model for Single Class and Single Scale Prediction
class YOLOvBIT(nn.Module):
    def __init__(self, num_classes=1):
        super(YOLOvBIT, self).__init__()
        self.conv1 = Conv(3, 80, 3, 2, 1)  # P1/2
        self.conv2 = Conv(80, 160, 3, 2, 1)  # P2/4
        self.c2f1 = C2f(160, 160, 3)  # num_bottlenecks: 3
        self.conv3 = Conv(160, 320, 3, 2, 1)  # P3/8
        self.c2f2 = C2f(320, 320, 6)  # num_bottlenecks: 6
        self.conv4 = Conv(320, 640, 3, 2, 1)  # P4/16
        self.c2f3 = C2f(640, 640, 6)  # num_bottlenecks: 6
        self.conv5 = Conv(640, 1280, 3, 2, 1)  # P5/32
        self.c2f4 = C2f(1280, 1280, 3)  # num_bottlenecks: 3
        self.sppf = SPPF(1280, 1280)  # P5/32
        self.detect = Detect(num_classes)  # Detection layer

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.c2f1(x2)
        x4 = self.conv3(x3)
        x5 = self.c2f2(x4)
        x6 = self.conv4(x5)
        x7 = self.c2f3(x6)
        x8 = self.conv5(x7)
        x9 = self.c2f4(x8)
        x10 = self.sppf(x9)
        outputs = self.detect(x10)
        return outputs

# Rest of your existing code
def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def bbox_ciou(box1, box2):
    """
    Returns the Complete IoU (CIoU) between box1 and box2.
    """
    # Calculate the IoU
    iou = bbox_iou(box1, box2, x1y1x2y2=False)
    
    # Calculate the center distance
    center_distance = torch.sum((box1[..., :2] - box2[..., :2]) ** 2, axis=-1)
    
    # Calculate the enclosing box
    enclose_x1 = torch.min(box1[..., 0] - box1[..., 2] / 2, box2[..., 0] - box2[..., 2] / 2)
    enclose_y1 = torch.min(box1[..., 1] - box1[..., 3] / 2, box2[..., 1] - box2[..., 3] / 2)
    enclose_x2 = torch.max(box1[..., 0] + box1[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2)
    enclose_y2 = torch.max(box1[..., 1] + box1[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2)
    enclose_diagonal = torch.sum((enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2, axis=-1)
    
    # Calculate the aspect ratio
    ar_gt = box2[..., 2] / (box2[..., 3] + 1e-16)  # Add small value to avoid division by zero
    ar_pred = box1[..., 2] / (box1[..., 3] + 1e-16)  # Add small value to avoid division by zero
    ar_loss = 4 / (torch.pi ** 2) * torch.pow(torch.atan(ar_gt) - torch.atan(ar_pred), 2)
    
    # Calculate the CIoU
    ciou = iou - (center_distance / (enclose_diagonal + 1e-16)) - ar_loss
    
    # Print intermediate values for debugging
    #print(f'IOU: {iou}')
    #print(f'Center Distance: {center_distance}')
    #print(f'Enclose Diagonal: {enclose_diagonal}')
    #print(f'AR Loss: {ar_loss}')
    #print(f'CIoU: {ciou}')
    
    return ciou

class YoloLoss(nn.Module):
    def __init__(self, lambda_box=7.5, lambda_obj=1.0, lambda_noobj=0.5, gamma=2.0, alpha=0.5):
        super(YoloLoss, self).__init__()
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.gamma = gamma
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, prediction, target):
        total_loss = 0
        batch_size = prediction.shape[0]
        #print(f'batch_size: {batch_size}')
        target = target.permute(0, 2, 3, 1)
        #print(f'target: {target.shape}')
        prediction = prediction.permute(0, 2, 3, 1)
        #print(f'prediction: {prediction.shape}')
        
        obj_mask = target[..., 4] > 0
        noobj_mask = target[..., 4] == 0

        coord_loss = torch.tensor(0.0, device=prediction.device)
        
        if obj_mask.sum() > 0:
            pred_box = prediction[obj_mask][:, :4]
            target_box = target[obj_mask][:, :4]
            
            #print(f"pred_box: {pred_box}")
            #print(f"target_box: {target_box}")

            ciou = bbox_ciou(pred_box, target_box)
            coord_loss = torch.mean(1 - ciou)

        obj_loss = torch.tensor(0.0, device=prediction.device)
        noobj_loss = torch.tensor(0.0, device=prediction.device)
        
        if obj_mask.sum() > 0:
            pred_obj = prediction[obj_mask][:, 4]
            target_obj = target[obj_mask][:, 4]
            obj_loss = self.bce_loss(pred_obj, target_obj)

            # Sigmoid 변환 후 값 출력
            sigmoid_pred_obj = torch.sigmoid(pred_obj)
            #print(f"Raw pred_obj: {pred_obj}")
            #print(f"Sigmoid pred_obj: {sigmoid_pred_obj}")

            obj_loss = self.alpha * (1 - sigmoid_pred_obj) ** self.gamma * obj_loss
            obj_loss = obj_loss.sum()

        if noobj_mask.sum() > 0:
            pred_noobj = prediction[noobj_mask][:, 4]
            target_noobj = target[noobj_mask][:, 4]
            noobj_loss = self.bce_loss(pred_noobj, target_noobj)

            # Sigmoid 변환 후 값 출력
            sigmoid_pred_noobj = torch.sigmoid(pred_noobj)
            #print(f"Raw pred_noobj: {pred_noobj}")
            #print(f"Sigmoid pred_noobj: {sigmoid_pred_noobj}")

            noobj_loss = self.alpha * sigmoid_pred_noobj ** self.gamma * noobj_loss
            noobj_loss = noobj_loss.sum()

        if obj_mask.sum() > 0:
            total_loss += (self.lambda_box * coord_loss + self.lambda_obj * obj_loss)
        total_loss += self.lambda_noobj * noobj_loss

        return total_loss / batch_size
    
# Dataset and DataLoader
class COCODataset(Dataset):
    def __init__(self, annotation_file, img_dir, S=20, C=1, max_bboxes=4, subset_size=10000):
        self.coco = COCO(annotation_file)
        self.img_dir = img_dir
        self.img_ids = self.coco.getImgIds()
        self.S = S
        self.C = C
        self.max_bboxes = max_bboxes
        self.person_class_id = 1  # person 클래스의 ID

        if subset_size is not None:
            self.img_ids = random.sample(self.img_ids, subset_size)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[self.person_class_id])  # person 클래스만 선택
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        if len(anns) == 0:
            return self.__getitem__((index + 1) % len(self.img_ids))

        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        original_width, original_height = image.shape[1], image.shape[0]

        bboxes = []
        class_labels = []

        for ann in anns:
            bbox = ann['bbox']
            x, y, width, height = bbox

            cx = (x + width / 2) / original_width
            cy = (y + height / 2) / original_height
            width /= original_width
            height /= original_height

            bboxes.append([cx, cy, width, height])
            class_labels.append(0)  # person 클래스의 index는 0으로 설정

        current_transform = get_train_transform()

        try:
            image, bboxes, class_labels = apply_transform(image, bboxes, class_labels, current_transform)
        except Exception as e:
            print(f"Error applying transform: {e}")
            return self.__getitem__((index + 1) % len(self.img_ids))

        target = np.zeros((5 + self.C, self.S, self.S))
        for bbox, label in zip(bboxes, class_labels):
            cx, cy, width, height = bbox
            cell_x = int(cx * self.S)
            cell_y = int(cy * self.S)

            if 0 <= cell_x < self.S and 0 <= cell_y < self.S:
                if target[4, cell_y, cell_x] == 0:
                    target[:4, cell_y, cell_x] = np.array([cx, cy, width, height])
                    target[4, cell_y, cell_x] = 1
                    target[5+label, cell_y, cell_x] = 1


        targets = torch.tensor(target, dtype=torch.float32)
        return image, targets  

#학습 조기 종료
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, train_loss):
        if self.best_loss is None:
            self.best_loss = train_loss
        elif train_loss < self.best_loss - self.min_delta:
            self.best_loss = train_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def reset(self):
        """
        조기 종료 상태를 초기화하는 메서드
        """
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

# Training and Evaluation Functions with Warmup and Warmup Momentum
def train(model, dataloader, optimizer, criterion, device, epoch, scheduler, early_stopping, save_path, warmup_iters=1000, warmup_ratio=0.1, warmup_momentum=0.8):
    model.train()
    total_loss = 0
    iters = len(dataloader)
    #warmup_steps = warmup_iters * iters

    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        # Warmup / 맨 처음 학습할 때만 활성화
        #current_iter = epoch * iters + batch_idx
        #if current_iter < warmup_steps:
            #lr_scale = (1 - warmup_ratio) * (current_iter / warmup_steps) + warmup_ratio
            #momentum_scale = (1 - warmup_momentum) * (current_iter / warmup_steps) + warmup_momentum
            #for pg in optimizer.param_groups:
                #pg['lr'] = lr_scale * 0.01
                #pg['momentum'] = momentum_scale * 0.937

        if batch_idx % 10 == 0:
            print(f'Train Epoch [{epoch+1}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

        if batch_idx % 10000 == 0 and batch_idx !=0 :
            print('Waiting for 1 minute before starting the next epoch.')
            time.sleep(60)  # 1 minute wait

    #if scheduler and epoch * iters >= warmup_steps:
        #scheduler.step()

    if scheduler:
        scheduler.step()

    avg_train_loss = total_loss / len(dataloader)
    
    # Early Stopping 체크
    #early_stopping(avg_train_loss)
    #if early_stopping.early_stop:
        #print(f"Early stopping at epoch {epoch+1}")
        #return avg_train_loss, True

    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    return avg_train_loss, False

def validate(model, dataloader, criterion, device, epoch, conf_threshold=0.5, iou_threshold=0.5):
    model.eval()
    total_loss = 0
    all_detections = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Apply NMS
            for i, output in enumerate(outputs):
                output = output.permute(0, 2, 3, 1)
                batch_size = output.shape[0]
                for b in range(batch_size):
                    pred_boxes = []
                    pred_scores = []
                    pred_labels = []
                    for y in range(output.shape[1]):
                        for x in range(output.shape[2]):
                            if output[b, y, x, 4] > conf_threshold:
                                pred_box = output[b, y, x, :4].cpu().numpy() * 640
                                score = output[b, y, x, 4].item()
                                label = output[b, y, x, 5:].argmax().item()
                                pred_boxes.append(pred_box)
                                pred_scores.append(score)
                                pred_labels.append(label)

                    if len(pred_boxes) > 0:
                        pred_boxes = torch.tensor(pred_boxes)
                        pred_scores = torch.tensor(pred_scores)
                        nms_indices = nms(pred_boxes, pred_scores, iou_threshold)
                        pred_boxes = pred_boxes[nms_indices]
                        pred_scores = pred_scores[nms_indices]
                        pred_labels = [pred_labels[i] for i in nms_indices]
                        all_detections.append((pred_boxes, pred_scores, pred_labels))
                    else:
                        all_detections.append((torch.tensor([]), torch.tensor([]), []))
                    
                    # Print confidence scores
                    for score in pred_scores:
                        print(f'Confidence score: {score.item()}')

            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    # Here, you can calculate additional metrics like mAP or recall using all_detections and all_targets

    return avg_loss, all_detections, all_targets

# Function to visualize a sample
def visualize_sample(dataset, index):
    image, target = dataset[index]
    image = image.permute(1, 2, 0).numpy() * 255  # Convert to HWC format

    fig, ax = plt.subplots(1)
    ax.imshow(image.astype(np.uint8))

    target = target.numpy()
    grid_size = target.shape[1]

    for y in range(grid_size):
        for x in range(grid_size):
            if target[4, y, x] > 0:  # Object present
                bx = target[0, y, x] * 640
                by = target[1, y, x] * 640
                bw = target[2, y, x] * 640
                bh = target[3, y, x] * 640
                rect = patches.Rectangle((bx - bw / 2, by - bh / 2), bw, bh, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
                class_id = np.argmax(target[5:, y, x])
                plt.text(bx, by, str(class_id), color='red', fontsize=8)

    plt.show()

#이미지 전처리1 
def get_train_transform():
    
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.5),
        #A.Lambda(image=safe_random_crop, p=0.5),
        A.GaussNoise(p=0.5),
        A.MotionBlur(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-10, 10), p=0.5),
        A.Resize(width=640, height=640), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

#이미지 전처리2 
def apply_transform(image, bboxes, class_labels, transform):
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    return transformed['image'], transformed['bboxes'], transformed['class_labels']

# Main
def main():
    train_annotation_file = r'D:\annotations\instances_train2017.json'
    train_img_dir = r'D:\train2017'
    val_annotation_file = r'D:\annotations\instances_val2017.json'
    val_img_dir = r'D:\val2017'

    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
    ])

    train_dataset = COCODataset(train_annotation_file, train_img_dir, subset_size=118287) #118287
    val_dataset = COCODataset(val_annotation_file, val_img_dir, subset_size=2000) #5000
    
    # Visualize a few samples
    for i in range(5):
        visualize_sample(train_dataset, i)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, persistent_workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 사용자 정의 YOLOv8 모델 로드
    model = YOLOvBIT(num_classes=1).to(device)  # 클래스 수를 1로 변경

    save_path = r'D:\VS_Code\@AI_Modeling\YOLOv8\yololv8_model_new_best_only_person_focalloss_AdamW_120_Copy.pt' #학습중인 모델 경로

    # 저장된 모델이 있으면 불러오기
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f'Model loaded from {save_path}')
        
    # Adjust hyperparameters based on the provided best values
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, betas=(0.9, 0.999), weight_decay=0.0005)
    criterion = YoloLoss(lambda_box=7.5, lambda_obj=1.0, lambda_noobj=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    num_epochs = 500
    warmup_iters = 1  # 조정할 수 있습니다.
    warmup_ratio = 0.1
    warmup_momentum = 0.8
    current_lr = 0.003  # 초기 학습률

    # Early Stopping 설정
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    for epoch in range(num_epochs):
        avg_train_loss, stop_training = train(model, train_dataloader, optimizer, criterion, device, epoch, scheduler, early_stopping, save_path, warmup_iters, warmup_ratio, warmup_momentum)
        print(f'Epoch {epoch+1}/{num_epochs}, Train AVG Loss: {avg_train_loss:.4f}')
        
        #val_loss, detections, targets = validate(model, val_dataloader, criterion, device, epoch)
        #print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

        #if stop_training:
            # 학습률 증가 및 조기 종료 상태 초기화
            #current_lr *= 2  # 학습률을 두 배로 증가
            #for param_group in optimizer.param_groups:
                #param_group['lr'] = current_lr
            #early_stopping.reset()
            #print(f"Early stopping at epoch {epoch+1}, but increasing learning rate to {current_lr} and continuing training.")
        
        print(f'Epoch {epoch+1} completed. Waiting for 1 minute before starting the next epoch.')
        time.sleep(60)  # 1 minute wait / 하드웨어 환경 한계로 인한 대기 시간

if __name__ == '__main__':
    main()