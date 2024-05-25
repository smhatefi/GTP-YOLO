import torch
import cv2
import matplotlib.pyplot as plt
from models.fire_detection_model import load_pretrained_model
from utils.general import non_max_suppression, scale_coords

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def detect_fire(image_path, model):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (640, 640))
    img_tensor = torch.from_numpy(img_resized).to(device).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Process the predictions and draw bounding boxes on the image
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=(0, 255, 0), line_thickness=2)

    return img


import numpy as np

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescales bounding box coordinates from the dimensions of img1_shape to img0_shape.
    Arguments:
    img1_shape: tuple, shape of the image used for inference (height, width)
    coords: tensor, bounding box coordinates (x1, y1, x2, y2)
    img0_shape: tuple, original shape of the image (height, width)
    ratio_pad: optional, padding ratio (if any)

    Returns:
    coords: tensor, rescaled bounding box coordinates
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = resized / original
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].round()

    # Clip boxes to image dimensions
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2

    return coords


# Example usage
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_pretrained_model()
    model.to(device)
    
    image_path = 'path/to/your/test_image.jpg'
    detected_img = detect_fire(image_path, model)

    # Convert BGR to RGB for displaying
    detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(detected_img_rgb)
    plt.axis('off')
    plt.show()
