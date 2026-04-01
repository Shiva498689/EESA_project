






# !pip install ultralytics sahi Installs YOLOv8 (ultralytics) 

import os           
import shutil       
import yaml         
import random       
import glob         
import xml.etree.ElementTree as ET
import cv2          
import torch        
import matplotlib.pyplot as plt
import seaborn as sns          
from ultralytics import YOLO   




USER_IMAGES_DIR = "/kaggle/input/pcb-defects/PCB_DATASET/images"
USER_ANNOTATIONS_DIR = "/kaggle/input/pcb-defects/PCB_DATASET/Annotations"



OUTPUT_DIR = "/kaggle/working/yolo_final_dataset"


DEVICE = 0 if torch.cuda.is_available() else 'cpu' 

EPOCHS = 30        
BATCH_SIZE = 8     
IMG_SIZE = 640      

print(f" HARDWARE CHECK: Running on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


def prepare_dataset():
  
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    
   
    for split in ['train', 'test']:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

    
    image_map = {}
    for root, dirs, files in os.walk(USER_IMAGES_DIR):
        for f in files:
           
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_map[f] = os.path.join(root, f)

 
    xml_list = []
    for root, dirs, files in os.walk(USER_ANNOTATIONS_DIR):
        for f in files:
            if f.endswith('.xml'):
                xml_list.append(os.path.join(root, f))
    
    print(f"   --> Found {len(image_map)} images and {len(xml_list)} XML annotations.")

  
    classes = set() 
    pairs = []      
    
    print(" Matching annotations to images...")
    for xml_path in xml_list:
     
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.iter('object'):
                classes.add(obj.find('name').text.strip()) 
        except: 
            continue 
       
        xml_fname = os.path.basename(xml_path)
        file_id = os.path.splitext(xml_fname)[0]
        
       
        found_img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            target_file = file_id + ext
            if target_file in image_map:
                found_img_path = image_map[target_file]
                break
        
      
        if found_img_path:
            pairs.append((xml_path, found_img_path))

   
    CLASSES = sorted(list(classes))
    print(f"   --> Detected Classes: {CLASSES}")
    print(f"   --> Successfully matched {len(pairs)} pairs.")

   
    
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.80)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    
    print(f" Splitting: {len(train_pairs)} Train images | {len(test_pairs)} Test images")

   
    def convert_and_save(pair_list, split_name):
        for xml_path, img_path in pair_list:
           
            fname = f"{os.path.basename(os.path.dirname(img_path))}_{os.path.basename(img_path)}"
            
         
            shutil.copy(img_path, f"{OUTPUT_DIR}/images/{split_name}/{fname}")
            
           
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
          
            size = root.find('size')
            w_img = int(size.find('width').text)
            h_img = int(size.find('height').text)
            
            # Create corresponding .txt file
            txt_path = f"{OUTPUT_DIR}/labels/{split_name}/{os.path.splitext(fname)[0]}.txt"
            
            with open(txt_path, 'w') as f:
                for obj in root.iter('object'):
                    c = obj.find('name').text.strip()
                    if c in CLASSES:
                        cid = CLASSES.index(c)
                        bnd = obj.find('bndbox')
                    
                        xmin = float(bnd.find('xmin').text)
                        xmax = float(bnd.find('xmax').text)
                        ymin = float(bnd.find('ymin').text)
                        ymax = float(bnd.find('ymax').text)
                 
                        xc = ((xmin + xmax) / 2) / w_img
                        yc = ((ymin + ymax) / 2) / h_img
                        w = (xmax - xmin) / w_img
                        h = (ymax - ymin) / h_img
                        
                        # Write to file: class_id x_center y_center width height
                        f.write(f"{cid} {xc} {yc} {w} {h}\n")

    # Execute the conversion for both splits
    convert_and_save(train_pairs, 'train')
    convert_and_save(test_pairs, 'test')

    yaml_data = {
        'train': f"{OUTPUT_DIR}/images/train",
        'val': f"{OUTPUT_DIR}/images/test", # Using test set as validation for simplicity
        'test': f"{OUTPUT_DIR}/images/test",
        'nc': len(CLASSES),     
        'names': CLASSES        
    }
    
    yaml_path = f"{OUTPUT_DIR}/data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)
        
    return yaml_path, CLASSES


def train_model(yaml_path):
    print("\n" + "="*40)
    print(" STARTING GPU TRAINING")
    print("="*40)
    
 
    model = YOLO('yolov8m.pt') 
    
    # Start Training
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,       
        batch=BATCH_SIZE,    
        device=DEVICE,     
        workers=4,           
        project='/kaggle/working/runs/detect',
        name='pcb_gpu_run', 
        verbose=True       
    )
    return model


def run_evaluation(model, yaml_path):
    print("\n" + "="*40)
    print(" CALCULATING METRICS ON TEST SET")
    print("="*40)
    
   
    metrics = model.val(data=yaml_path, split='test', device=DEVICE)
    
  
    p = metrics.results_dict['metrics/precision(B)']
    r = metrics.results_dict['metrics/recall(B)']
    map50 = metrics.results_dict['metrics/mAP50(B)'] 
    
    f1 = 2 * (p * r) / (p + r + 1e-6) # 1e-6 added to avoid division by zero
    
    print(f"\n FINAL TEST RESULTS:")
    print(f"    Precision: {p:.2%}")
    print(f"    Recall:    {r:.2%}")
    print(f"    F1 Score:  {f1:.2f}")
    print(f"    mAP@50:    {map50:.2%}")

    # Display Confusion Matrix

    cm_path = f"/kaggle/working/runs/detect/pcb_gpu_run/confusion_matrix.png"
    if os.path.exists(cm_path):
        plt.figure(figsize=(8, 6))
        img = cv2.imread(cm_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Convert BGR (OpenCV standard) to RGB (Matplotlib standard)
        plt.axis('off')
        plt.title("Confusion Matrix")
        plt.show()



  def visualize_results(model, class_names):
    """
    Visualizes the performance by plotting:
    LEFT: The image with Ground Truth (Manual Annotation)
    RIGHT: The image with Model Prediction (AI Output)
    """
    print("\n" + "="*40)
    print(" VISUALIZATION: ACTUAL (Left) vs PREDICTED (Right)")
    print("="*40)
    
    test_img_dir = f"{OUTPUT_DIR}/images/test"
    test_lbl_dir = f"{OUTPUT_DIR}/labels/test"
    
    # Get list of all test images
    test_images = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
    
    # Pick 5 random samples
    samples = random.sample(test_images, min(5, len(test_images)))
    
    for img_file in samples:
        img_path = os.path.join(test_img_dir, img_file)
        lbl_path = os.path.join(test_lbl_dir, os.path.splitext(img_file)[0] + ".txt")
        
        # Load Image using OpenCV
        img_raw = cv2.imread(img_path)
        h, w, _ = img_raw.shape 
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        
     
        img_actual = img_rgb.copy()
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    cls_id = int(data[0])
                    # Read normalized YOLO coordinates
                    xc, yc, nw, nh = map(float, data[1:])
                    
                  
                    # Convert back to x_min, y_min, x_max, y_max for drawing
                    x1 = int((xc - nw/2) * w)
                    y1 = int((yc - nh/2) * h)
                    x2 = int((xc + nw/2) * w)
                    y2 = int((yc + nh/2) * h)
                    
                    # Draw Green Box (Ground Truth)
                    cv2.rectangle(img_actual, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Draw Label Text
                    label = class_names[cls_id]
                    cv2.putText(img_actual, f"ACTUAL: {label}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
       
        results = model.predict(img_path, conf=0.25, device=DEVICE, verbose=False)
        
       
        res_plot = results[0].plot() 
        img_pred = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)
   
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        
        axes[0].imshow(img_actual)
        axes[0].set_title(" ACTUAL ANNOTATIONS (Ground Truth)", color='green', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(img_pred)
        axes[1].set_title(" MODEL PREDICTION (AI Result)", color='blue', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()


 if __name__ == "__main__":
 
    data_yaml, classes = prepare_dataset()
    
    model = train_model(data_yaml)
    
    run_evaluation(model, data_yaml)
    
    visualize_results(model, classes)
