from PIL import Image
from IPython.display import display
from IPython.display import clear_output
import cv2
import numpy as np
from pathlib import Path

def convert_pil(im, bgr=True):
    if isinstance(im, np.ndarray):
        if bgr:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
    elif isinstance(im, Image.Image):
        pass
    elif isinstance(im, str):
        im = Image.open(im)
    else:
        raise ValueError(f"Invalid data type. {type(im)}")

    return im

def convert_array(im):
    if isinstance(im, np.ndarray):
        pass
    elif isinstance(im, Image.Image):
        im = np.array(im)
    elif isinstance(im, str):
        im = cv2.imread(im)
    else:
        raise ValueError(f"Invalid data type. {type(im)}")

    return im   

# 画像を表示（PIL or numpy array(opencv) or file path）
def show_image(im, bgr=True, handle=None):
    im = convert_pil(im)

    if handle is None:
        display(im)
    else:
        handle.update(im)

# 複数の画像を表示（PIL or numpy array(opencv) or file path）
# titlesにいれた文字列をタイトルとして表示
# titlesの値がNoneの場合はfile pathの場合はファイル名を表示、それ以外（PIL or numpy arrya）の場合はタイトルを表示しない。   
def show_images(im_list, titles=[], bgr=True):    
    titles += [None] * len(im_list)
    for im, title in zip(im_list, titles):
        if title is None:
            if isinstance(im, str):
                title = Path(im).name
            else:
                title = ""
            
        print(title)
        show_image(im, bgr=bgr)

def show_multi_images(im_list_col, bgr=True):
    n = None
    for im_list in im_list_col:
        if n is None:
            n = len(im_list)
        else:
            assert n == len(im_list), f"The number of images must be the same. {n}!={len(im_list)}"
            n = len(im_list)

    images_rows = list(zip(*im_list_col))
    
    for images in images_rows:
        images = [convert_array(im) for im in images]
        
        h = min(im.shape[0] for im in images)
        images = [cv2.resize(im, (im.shape[1], h)) for im in images]
        stack_iamge = np.hstack(images)
        
        show_image(stack_iamge, bgr=bgr)

# 指定ディレクトリの画像一覧を取得する
def get_image_path_list(root_path, recursive=False, exts=None, pathlib=False):
    root_path = Path(root_path)
    
    if exts is None:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    else:
        exts = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in exts}
   
    if recursive == True:
        image_path_list = [f for f in root_path.rglob('*') if f.is_file() and f.suffix.lower() in exts]
    else:
        image_path_list = [f for f in root_path.iterdir() if f.is_file() and f.suffix.lower() in exts]

    if not pathlib:
        image_path_list = [str(p) for p in image_path_list]

    return image_path_list
    