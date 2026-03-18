import cv2
import numpy as np
import logging

logger = logging.getLogger()

def generate_heatmap(anomaly_map, normalize=True, cut_th=None, bgr=True):
    anomaly_map = anomaly_map.squeeze()
    
    if cut_th is None:
        mask = None
    else:
        mask = anomaly_map < cut_th

    if normalize:
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    
    anomaly_map = (anomaly_map * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)

    if bgr:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2BGRA)
    else:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGRA)

    return heatmap, mask

def overlay_heatmap(im_base, anomaly_map, normalize=True, alpha=0.5, cut_th=None, bgr=True):
    im_heatmap, mask = generate_heatmap(anomaly_map, normalize, cut_th, bgr)
    im_overlay = overlay(im_base, im_heatmap, alpha, mask)

    return im_overlay

def overlay(im_base, im_overlay, alpha=0.5, mask=None):
    h, w = im_base.shape[:2]

    im_overlay = cv2.resize(im_overlay, (w, h))
    im_overlay_rgb = im_overlay[:, :, :3]
    im_alpha = (im_overlay[:, :, 3:4] / 255.0) * alpha

    if mask is not None:
        mask = cv2.resize(mask.astype(np.uint8), (w, h)).astype(bool)
        im_alpha[mask] = 0.0

    im_overlay= (im_overlay_rgb * im_alpha + im_base * (1 - im_alpha)).astype("uint8")
    
    return im_overlay

def generate_mask(anomaly_map, threshold=0.5, kernel_size=4):
    anomaly_map = anomaly_map.squeeze()
    mask = np.zeros_like(anomaly_map, dtype=np.uint8)
    mask[anomaly_map > threshold] = 255

    l = np.arange(-kernel_size, kernel_size + 1)
    x, y = np.meshgrid(l, l)
    kernel = np.array((x**2 + y**2) <= kernel_size**2, dtype=np.uint8)

    im_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return im_opened

def overlay_mask_edge(
        im_base,
        anomaly_map,
        threshold=0.5,
        kernel_size=4,
        color=(0, 0, 255),
        backcolor=None,
        thickness=1,
        anti_aliasing=True,
    ):
    h, w = im_base.shape[:2]

    anomaly_map = anomaly_map.squeeze()

    mask = generate_mask(anomaly_map, threshold, kernel_size)
    mask = cv2.resize(mask, (w, h))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    im_overlay = im_base.copy()

    line_type = cv2.LINE_AA if anti_aliasing else cv2.LINE_8

    if backcolor is not None:
        cv2.drawContours(im_overlay, contours, -1, backcolor, thickness=thickness+1, lineType=line_type)
    cv2.drawContours(im_overlay, contours, -1, color, thickness=thickness, lineType=line_type)

    return im_overlay
