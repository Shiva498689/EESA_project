# ==============================================================================
# 📋 PCB DEFECT CLASSIFICATION PIPELINE - DETAILED ANALYSIS
# ==============================================================================
# This script converts a dataset of PCB images and XML annotations (Pascal VOC format)
# into a format compatible with YOLOv8, trains a model, and visualizes the results.

# 📦 LIBRARY IMPORTS
# ------------------
# !pip install ultralytics sahi  <-- Installs YOLOv8 (ultralytics) and Sahi (slicing aid, though not used here directly)

import os           # For interacting with the Operating System (file paths, making directories)
import shutil       # High-level file operations (copying images, deleting folders)
import yaml         # For reading/writing YAML configuration files (required by YOLO)
import random       # For shuffling data to ensure random splitting of Train/Test sets
import glob         # For finding pathnames matching a pattern (like *.jpg)
import xml.etree.ElementTree as ET # For parsing XML files (Legacy annotation format)
import cv2          # OpenCV: The industry standard for Image Processing (loading, drawing boxes)
import torch        # PyTorch: The deep learning framework powering YOLOv8
import matplotlib.pyplot as plt # For plotting graphs and displaying images
import seaborn as sns           # For making statistical plots prettier (optional here but good practice)
from ultralytics import YOLO    # The main class to load and train YOLOv8 models

# ==========================================
# ⚙️ CONFIGURATION (USER INPUTS)
# ==========================================
# 1. INPUT PATHS
# These paths point to where your raw data sits in the Kaggle environment.
USER_IMAGES_DIR = "/kaggle/input/pcb-defects/PCB_DATASET/images"
USER_ANNOTATIONS_DIR = "/kaggle/input/pcb-defects/PCB_DATASET/Annotations"

# 2. OUTPUT PATH
# This is where we will generate the "Clean" dataset.
# Kaggle's /input directory is Read-Only. We must write to /working.
OUTPUT_DIR = "/kaggle/working/yolo_final_dataset"

# 3. TRAINING SETTINGS
# Device Selection: Checks if an NVIDIA GPU is available via CUDA.
# If yes, it uses device '0' (the first GPU). If no, it falls back to 'cpu'.
# precise Hardware acceleration is critical for Object Detection.
DEVICE = 0 if torch.cuda.is_available() else 'cpu' 

EPOCHS = 30         # How many times the model sees the entire dataset.
BATCH_SIZE = 8      # How many images are processed at once before updating model weights.
IMG_SIZE = 640      # Resolution to resize images to. YOLO is optimized for 640x640.

print(f"🚀 HARDWARE CHECK: Running on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ==========================================
# 🛠️ PART 1: PREPARE DATA & SPLIT
# ==========================================
def prepare_dataset():
    """
    Converts XML (Pascal VOC) annotations to TXT (YOLO) format and organizes folders.
    YOLO requires a very specific folder structure:
        /images/train, /images/test
        /labels/train, /labels/test
    """
    # Safety Check: If the output directory exists from a previous run, delete it.
    # This ensures we don't mix old data with new data (Data Leakage prevention).
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    
    # Create the directory structure required by YOLO
    for split in ['train', 'test']:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

    print("\n🔍 Indexing files...")
    
    # 1. Map all images
    # We create a dictionary (hash map) for O(1) lookup speeds.
    # Key: Filename (e.g., 'image_01.jpg'), Value: Full Path
    image_map = {}
    for root, dirs, files in os.walk(USER_IMAGES_DIR):
        for f in files:
            # Check for common image extensions
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_map[f] = os.path.join(root, f)

    # 2. Map all XMLs
    # We scan the annotations folder for .xml files.
    xml_list = []
    for root, dirs, files in os.walk(USER_ANNOTATIONS_DIR):
        for f in files:
            if f.endswith('.xml'):
                xml_list.append(os.path.join(root, f))
    
    print(f"   --> Found {len(image_map)} images and {len(xml_list)} XML annotations.")

    # 3. Pair XMLs with Images and Find Classes
    classes = set() # A Set automatically removes duplicates, perfect for collecting class names.
    pairs = []      # Will hold tuples of (path_to_xml, path_to_image)
    
    print("🔗 Matching annotations to images...")
    for xml_path in xml_list:
        # Extract Classes strictly to build the 'names' list for YOLO
        try:
            tree = ET.parse(xml_path) # Parse the XML tree structure
            root = tree.getroot()
            for obj in root.iter('object'):
                classes.add(obj.find('name').text.strip()) # Add class name to set
        except: 
            continue # If XML is corrupt, skip it

        # Find the corresponding Image
        # XML filename usually matches Image filename (e.g., img1.xml matches img1.jpg)
        xml_fname = os.path.basename(xml_path)
        file_id = os.path.splitext(xml_fname)[0] # Remove .xml extension
        
        # Robust Logic: Try to find the file_id with ANY valid image extension
        found_img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            target_file = file_id + ext
            if target_file in image_map:
                found_img_path = image_map[target_file]
                break
        
        # Only add to dataset if both XML and Image exist
        if found_img_path:
            pairs.append((xml_path, found_img_path))

    # Sort classes alphabetically to ensure consistent ID mapping (0, 1, 2...)
    CLASSES = sorted(list(classes))
    print(f"   --> Detected Classes: {CLASSES}")
    print(f"   --> Successfully matched {len(pairs)} pairs.")

    # 4. Split Data (80% Train, 20% Test)
    # Random shuffle is CRITICAL to avoid bias (e.g., if all defects of one type are at the end)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.80)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    
    print(f"📦 Splitting: {len(train_pairs)} Train images | {len(test_pairs)} Test images")

    # 5. Conversion Function (The Core Logic)
    def convert_and_save(pair_list, split_name):
        for xml_path, img_path in pair_list:
            # Generate a unique filename to prevent overwriting if names are generic (like '1.jpg')
            fname = f"{os.path.basename(os.path.dirname(img_path))}_{os.path.basename(img_path)}"
            
            # Copy the actual image file to the destination folder
            shutil.copy(img_path, f"{OUTPUT_DIR}/images/{split_name}/{fname}")
            
            # Parse XML & Write Label
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get Image Dimensions (Needed for Normalization)
            size = root.find('size')
            w_img = int(size.find('width').text)
            h_img = int(size.find('height').text)
            
            # Create corresponding .txt file
            txt_path = f"{OUTPUT_DIR}/labels/{split_name}/{os.path.splitext(fname)[0]}.txt"
            
            with open(txt_path, 'w') as f:
                for obj in root.iter('object'):
                    c = obj.find('name').text.strip()
                    if c in CLASSES:
                        cid = CLASSES.index(c) # Convert Class Name to Integer ID (0, 1, 2)
                        bnd = obj.find('bndbox')
                        
                        # Extract Raw Coordinates (Pixel values)
                        xmin = float(bnd.find('xmin').text)
                        xmax = float(bnd.find('xmax').text)
                        ymin = float(bnd.find('ymin').text)
                        ymax = float(bnd.find('ymax').text)
                        
                        # ---------------------------------------------------------
                        # 🧮 MATH: COORDINATE NORMALIZATION
                        # XML gives top-left (xmin, ymin) and bottom-right (xmax, ymax).
                        # YOLO requires Center (x,y) and Width/Height, normalized 0-1.
                        #
                        # 1. Center X = (xmin + xmax) / 2
                        # 2. Normalize Center X = Center X / Image_Width
                        # 3. Width = (xmax - xmin) / Image_Width
                        # ---------------------------------------------------------
                        xc = ((xmin + xmax) / 2) / w_img
                        yc = ((ymin + ymax) / 2) / h_img
                        w = (xmax - xmin) / w_img
                        h = (ymax - ymin) / h_img
                        
                        # Write to file: class_id x_center y_center width height
                        f.write(f"{cid} {xc} {yc} {w} {h}\n")

    # Execute the conversion for both splits
    convert_and_save(train_pairs, 'train')
    convert_and_save(test_pairs, 'test')

    # 6. Create YAML Configuration
    # This file tells YOLO where the images are and what the class names are.
    yaml_data = {
        'train': f"{OUTPUT_DIR}/images/train",
        'val': f"{OUTPUT_DIR}/images/test", # Using test set as validation for simplicity
        'test': f"{OUTPUT_DIR}/images/test",
        'nc': len(CLASSES),     # Number of Classes
        'names': CLASSES        # List of Class Names
    }
    
    yaml_path = f"{OUTPUT_DIR}/data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)
        
    return yaml_path, CLASSES

# ==========================================
# 🔥 PART 2: GPU TRAINING
# ==========================================
def train_model(yaml_path):
    print("\n" + "="*40)
    print("🔥 STARTING GPU TRAINING")
    print("="*40)
    
    # Load the Model
    # 'yolov8m.pt' -> 'm' stands for Medium. 
    # Sizes: n (nano), s (small), m (medium), l (large), x (extra large).
    # Medium is a good balance between speed and accuracy for PCBs.
    model = YOLO('yolov8m.pt') 
    
    # Start Training
    model.train(
        data=yaml_path,
        epochs=EPOCHS,        # Total training iterations
        imgsz=IMG_SIZE,       # Resize images to 640x640
        batch=BATCH_SIZE,     # Number of images per GPU cycle
        device=DEVICE,        # Enforce GPU (cuda:0)
        workers=4,            # Number of CPU threads to load data (prevents bottlenecks)
        project='/kaggle/working/runs/detect', # Where to save logs/weights
        name='pcb_gpu_run',   # Name of the subfolder for this specific run
        verbose=True          # Print detailed logs
    )
    return model

# ==========================================
# 📊 PART 3: TEST SET EVALUATION
# ==========================================
def run_evaluation(model, yaml_path):
    print("\n" + "="*40)
    print("🧪 CALCULATING METRICS ON TEST SET")
    print("="*40)
    
    # Run validation strictly on the 'test' split defined in yaml
    # This uses the trained weights to predict on unseen data.
    metrics = model.val(data=yaml_path, split='test', device=DEVICE)
    
    # Extract Key Metrics
    # Precision: Accuracy of positive predictions (Low False Positives)
    # Recall: Ability to find all positives (Low False Negatives)
    p = metrics.results_dict['metrics/precision(B)']
    r = metrics.results_dict['metrics/recall(B)']
    map50 = metrics.results_dict['metrics/mAP50(B)'] # Mean Average Precision at IoU 0.5
    
    # Calculate F1 Score (Harmonic Mean of Precision and Recall)
    f1 = 2 * (p * r) / (p + r + 1e-6) # 1e-6 added to avoid division by zero
    
    print(f"\n✅ FINAL TEST RESULTS:")
    print(f"   🎯 Precision: {p:.2%}")
    print(f"   🔍 Recall:    {r:.2%}")
    print(f"   ⚖️ F1 Score:  {f1:.2f}")
    print(f"   🏆 mAP@50:    {map50:.2%}")

    # Display Confusion Matrix
    # YOLO automatically generates this matrix during validation. We just load and show it.
    cm_path = f"/kaggle/working/runs/detect/pcb_gpu_run/confusion_matrix.png"
    if os.path.exists(cm_path):
        plt.figure(figsize=(8, 6))
        img = cv2.imread(cm_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Convert BGR (OpenCV standard) to RGB (Matplotlib standard)
        plt.axis('off')
        plt.title("Confusion Matrix")
        plt.show()

# ==========================================
# 👁️ PART 4: SIDE-BY-SIDE VISUALIZATION
# ==========================================
def visualize_results(model, class_names):
    """
    Visualizes the performance by plotting:
    LEFT: The image with Ground Truth (Manual Annotation)
    RIGHT: The image with Model Prediction (AI Output)
    """
    print("\n" + "="*40)
    print("👁️ VISUALIZATION: ACTUAL (Left) vs PREDICTED (Right)")
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
        h, w, _ = img_raw.shape # Get dimensions to un-normalize coordinates later
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        
        # --- LEFT: DRAW ACTUAL DEFECTS (FROM ANNOTATIONS) ---
        img_actual = img_rgb.copy()
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    cls_id = int(data[0])
                    # Read normalized YOLO coordinates
                    xc, yc, nw, nh = map(float, data[1:])
                    
                    # 🧮 MATH: UN-NORMALIZATION (YOLO -> PIXELS)
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
        
        # --- RIGHT: DRAW PREDICTED DEFECTS (FROM MODEL) ---
        # Run inference on the image
        # conf=0.25: Only show boxes where model is >25% confident
        results = model.predict(img_path, conf=0.25, device=DEVICE, verbose=False)
        
        # results[0].plot() creates an image with boxes already drawn by Ultralytics
        res_plot = results[0].plot() 
        img_pred = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)
        
        # --- PLOT SIDE BY SIDE ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        
        axes[0].imshow(img_actual)
        axes[0].set_title("✅ ACTUAL ANNOTATIONS (Ground Truth)", color='green', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(img_pred)
        axes[1].set_title("🤖 MODEL PREDICTION (AI Result)", color='blue', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

# ==========================================
# 🚀 MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # Step 1: Prepare the dataset (Convert XML -> YOLO Txt)
    data_yaml, classes = prepare_dataset()
    
    # Step 2: Train the YOLOv8 Medium model on the GPU
    model = train_model(data_yaml)
    
    # Step 3: Calculate mathematical metrics (mAP, F1)
    run_evaluation(model, data_yaml)
    
    # Step 4: Visual verification (Show images)
    visualize_results(model, classes)
