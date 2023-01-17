import cv2
import numpy as np


def pad(img, size):
    ratio = size[0] / size[1]
    h, w = img.shape[:2]
    if w / h < ratio:
        delta_w = h * ratio - w
        delta_h = 0
    else:
        delta_w = 0
        delta_h = w / ratio - h

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    resized = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
        value=color)
    return resized

def img2roi(img, is_dicom=False):
    """
    Returns ROI area in other words 
    cuts the image to a desired one

    Because there are machine label tags,
    undesired details out of the breast image.
    """
    if not is_dicom:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    img = np.array(img * 255, dtype = np.uint8)
    bin_img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)

    ys = contour.squeeze()[:, 0]
    xs = contour.squeeze()[:, 1]
    roi =  img[np.min(xs):np.max(xs), np.min(ys):np.max(ys)]
    
    return roi

# def convert_dcm_to_png(image_path, size, output_image_dir):
#     patient_id = image_path.parent.name
#     image_id = image_path.stem

#     dicom = dicomsdl.open(str(image_path))
#     img = dicom.pixelData()

#     img = (img - img.min()) / (img.max() - img.min())

#     if dicom.getPixelDataInfo()["PhotometricInterpretation"] == "MONOCHROME1":
#         img = 1 - img

#     img = cv2.resize(img, (size, size))

#     output_image_path = output_image_dir / f"{patient_id}_{image_id}.png"
#     cv2.imwrite(str(output_image_path), (img * 255).astype(np.uint8))