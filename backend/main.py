from google.colab import drive
drive.mount('/content/drive')



# @title Gun & Knife Detection & Tracking API for Colab - WITH FRAME SAVING
# Run this entire notebook to set up the API server with your detection models

!pip install -q fastapi uvicorn nest-asyncio python-multipart aiofiles tensorflow tensorflow-hub ultralytics opencv-python-headless

import os
import uuid
import shutil
import threading
import time
import subprocess
import traceback
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO

# Apply nest_asyncio
nest_asyncio.apply()

# Create directories
UPLOAD_DIR = Path("/content/uploads")
OUTPUT_DIR = Path("/content/outputs")  # Processed video outputs
FRAMES_DIR = Path("/content/frames")  # ALL incoming frames from webcam
DETECTED_IMAGES_DIR = Path("/content/detected_images")  # Only gun/knife-detected images

# Create all directories
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
DETECTED_IMAGES_DIR.mkdir(exist_ok=True)

print(f"✅ Directories created:")
print(f"   - Frames directory: {FRAMES_DIR}")
print(f"   - Detected images: {DETECTED_IMAGES_DIR}")
print(f"   - Output videos: {OUTPUT_DIR}")

# Session storage for webcam recording and tracking
active_sessions = {}
recording_buffers = {}
tracked_holders_by_session = {}  # Store tracked holders per session

# ============================================
# LOAD YOUR MODELS
# ============================================

print("\nLoading models...")

# Store model info for debugging
model_info = {}

# Load gun detection models - UPDATE THESE PATHS TO YOUR ACTUAL MODEL PATHS
gun_models = []
model_paths = [
    '/content/drive/MyDrive/Colab_Projects/best.pt',
    '/content/drive/MyDrive/Colab_Projects/best (3).pt',
    '/content/drive/MyDrive/Colab_Projects/best (2).pt',
    '/content/drive/MyDrive/Colab_Projects/model.pt',
    '/content/drive/MyDrive/Colab_Projects/best (4).pt'
]

for model_path in model_paths:
    try:
        if os.path.exists(model_path):
            model = YOLO(model_path)
            gun_models.append(model)
            # Store model labels for debugging
            if hasattr(model, 'model') and hasattr(model.model, 'names'):
                model_info[model_path] = model.model.names
            print(f"✓ Loaded model: {model_path}")
        else:
            print(f"⚠️ Model not found: {model_path}")
    except Exception as e:
        print(f"✗ Error loading {model_path}: {e}")

# Load person detection model
person_model_path = "/content/drive/MyDrive/Colab_Projects/yolo_human.pt"
if os.path.exists(person_model_path):
    person_model = YOLO(person_model_path)
    print(f"✓ Loaded person model: {person_model_path}")
else:
    print("⚠️ Person model not found, using default")
    person_model = YOLO("yolov8n.pt")

# Load pose model
try:
    pose_model = YOLO("yolov8n-pose.pt")
    print("✓ Loaded pose model")
except:
    pose_model = None
    print("⚠️ Pose model not loaded")

print(f"✓ Total gun models loaded: {len(gun_models)}")

# Print label mappings for debugging
print("\n📋 Model Label Mappings:")
for path, labels in model_info.items():
    print(f"   {path.split('/')[-1]}: {labels}")

# ============================================
# DETECTION FUNCTIONS
# ============================================

def inside_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    return inter / (area1 + area2 - inter + 1e-6)


def dist_to_center(pt, box):
    cx, cy = box_center(box)
    return np.linalg.norm(np.array(pt) - np.array([cx, cy]))


def merge_boxes(b1, b2):
    return [
        int(min(b1[0], b2[0])),
        int(min(b1[1], b2[1])),
        int(max(b1[2], b2[2])),
        int(max(b1[3], b2[3]))
    ]


# ------------------ Detection Functions ------------------

def detect_weapons(frame, conf=0.6):
    """Detect both guns and knives from the same models based on label mappings"""
    gun_candidates = []
    knife_candidates = []

    # Define weapon categories based on labels
    gun_labels = {'gun', 'guns', 'pistol', 'handgun', 'rifle'}
    knife_labels = {'knife'}
    grenade_labels = {'grenade'}  # Optional, can be treated separately if needed

    for model in gun_models:
        res = model(frame, conf=conf, verbose=False)[0]

        if res.boxes is None:
            continue

        # Get model's class names
        if hasattr(model, 'names'):
            class_names = model.names
        else:
            class_names = {}

        for box in res.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])

            # Get class name (convert to lowercase for comparison)
            class_name = class_names.get(cls, str(cls)).lower()

            # Categorize based on class name
            if class_name in knife_labels:
                knife_candidates.append(([x1, y1, x2, y2], score, class_name))
            elif class_name in gun_labels:
                gun_candidates.append(([x1, y1, x2, y2], score, class_name))
            # Optional: Handle grenades or other weapons
            elif class_name in grenade_labels:
                # Treat as gun for now, or create separate category
                gun_candidates.append(([x1, y1, x2, y2], score, class_name))

    # Process gun detections with NMS
    if gun_candidates:
        gun_candidates.sort(key=lambda x: x[1], reverse=True)
        final_guns = []
        used = [False] * len(gun_candidates)

        for i, (box_i, _, _) in enumerate(gun_candidates):
            if used[i]:
                continue
            final_guns.append(box_i)
            for j, (box_j, _, _) in enumerate(gun_candidates):
                if i != j and not used[j]:
                    if iou(box_i, box_j) > 0.4:
                        used[j] = True
    else:
        final_guns = []

    # Process knife detections with NMS
    if knife_candidates:
        knife_candidates.sort(key=lambda x: x[1], reverse=True)
        final_knives = []
        used = [False] * len(knife_candidates)

        for i, (box_i, _, _) in enumerate(knife_candidates):
            if used[i]:
                continue
            final_knives.append(box_i)
            for j, (box_j, _, _) in enumerate(knife_candidates):
                if i != j and not used[j]:
                    if iou(box_i, box_j) > 0.4:
                        used[j] = True
    else:
        final_knives = []

    return final_guns, final_knives

# Keep original function names for backward compatibility
def detect_guns(frame, conf=0.6):
    guns, _ = detect_weapons(frame, conf)
    return guns

def detect_knives(frame, conf=0.6):
    _, knives = detect_weapons(frame, conf)
    return knives

def detect_humans(frame, conf=0.4):

    res = person_model(frame, conf=conf, verbose=False)[0]

    human_boxes=[]

    if res.boxes is None:
        return human_boxes


    for box in res.boxes:

        x1,y1,x2,y2 = map(int, box.xyxy[0])
        human_boxes.append([x1,y1,x2,y2])

    return human_boxes



# ------------------ Pose & Wrist Logic ------------------

def get_wrists_from_pose(crop, offset_x, offset_y, conf_thresh=0.3):

    wrists=[]

    res = pose_model(crop, verbose=False)[0]

    if res.keypoints is None or res.keypoints.xy is None:
        return wrists

    if len(res.keypoints.xy)==0:
        return wrists


    kpts_all = res.keypoints.xy
    conf_all = res.keypoints.conf


    best_idx = np.argmax(conf_all.mean(axis=1))

    kpts = kpts_all[best_idx]
    conf = conf_all[best_idx]


    for idx in [9,10]:

        if conf[idx] >= conf_thresh:

            x = int(kpts[idx][0] + offset_x)
            y = int(kpts[idx][1] + offset_y)

            wrists.append((x,y,conf[idx]))


    return wrists



def fallback_score(human_box, weapon_box):

    iou_score = iou(human_box, weapon_box)

    hc = box_center(human_box)
    wc = box_center(weapon_box)

    dist_score = 1.0/(np.linalg.norm(np.array(hc)-np.array(wc))+1.0)

    return 0.6*iou_score + 0.4*dist_score



# ------------------ Association ------------------

def find_weapon_holder(frame, human_boxes, weapon_boxes, weapon_type="gun"):

    associations=[]

    for weapon_box in weapon_boxes:

        gx1,gy1,gx2,gy2 = weapon_box
        weapon_area = (gx2-gx1)*(gy2-gy1)

        # -------- New Logic : discard fake weapons --------
        valid = False
        for hbox in human_boxes:

            hx1,hy1,hx2,hy2 = hbox
            human_area = (hx2-hx1)*(hy2-hy1)

            if weapon_area < human_area:   # weapon must be smaller than human
                valid = True
                break

        if not valid:
            continue
        # -----------------------------------------------

        wrist_candidates=[]

        for hbox in human_boxes:

            x1,y1,x2,y2 = hbox

            crop = frame[y1:y2, x1:x2]

            wrists = get_wrists_from_pose(crop, x1, y1)

            for wrist in wrists:

                wx,wy,conf = wrist

                if inside_box(wx,wy,weapon_box):

                    d = dist_to_center((wx,wy), weapon_box)

                    wrist_candidates.append((d,wrist,hbox))

        if wrist_candidates:

            wrist_candidates.sort(key=lambda x:x[0])

            _,best_wrist,best_human = wrist_candidates[0]

            associations.append({

                "weapon_box":weapon_box,
                "weapon_type": weapon_type,
                "human_box":best_human,
                "merged_box":merge_boxes(best_human,weapon_box),
                "wrist":best_wrist,
                "all_wrists":[w[1] for w in wrist_candidates]

            })

            continue

        best_human=None
        best_score=0

        for hbox in human_boxes:

            score = fallback_score(hbox, weapon_box)

            if score > best_score:

                best_score=score
                best_human=hbox

        if best_human is not None:

            associations.append({

                "weapon_box":weapon_box,
                "weapon_type": weapon_type,
                "human_box":best_human,
                "merged_box":merge_boxes(best_human,weapon_box),
                "wrist":None,
                "all_wrists":[]

            })

    return associations

def find_gun_holder(frame, human_boxes, gun_boxes):
    return find_weapon_holder(frame, human_boxes, gun_boxes, "gun")

def find_knife_holder(frame, human_boxes, knife_boxes):
    return find_weapon_holder(frame, human_boxes, knife_boxes, "knife")

# ============================================
# TRACKING FUNCTIONS
# ============================================

def update_tracking(tracked_holders, humans):
    """Update tracked holders based on IoU matching"""
    updated_holders = []

    for holder in tracked_holders:
        best_iou = 0
        best_human = None

        for hbox in humans:
            score = iou(holder, hbox)
            if score > best_iou:
                best_iou = score
                best_human = hbox

        if best_human is not None and best_iou > 0.3:
            updated_holders.append(best_human)

    return updated_holders

def draw_annotations(frame, guns, knives, tracked_holders, gun_associations, knife_associations):
    """Draw all annotations on the frame"""
    img_with_boxes = frame.copy()

    # Draw weapon holders (tracked holders) - YELLOW
    for holder in tracked_holders:
        x1, y1, x2, y2 = holder
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(img_with_boxes, "Tracked Weapon Holder", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Draw guns - RED
    for gun in guns:
        x1, y1, x2, y2 = gun
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_with_boxes, "GUN", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw knives - ORANGE
    for knife in knives:
        x1, y1, x2, y2 = knife
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(img_with_boxes, "KNIFE", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Draw gun associations (merged boxes) - GREEN if wrist detected
    for assoc in gun_associations:
        if assoc.get("merged_box"):
            x1, y1, x2, y2 = assoc["merged_box"]
            color = (0, 255, 0) if assoc.get("wrist") else (255, 0, 255)
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_with_boxes, f"GUN HOLDER", (x1, y1-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if assoc.get("wrist"):
                wx, wy, _ = assoc["wrist"]
                cv2.circle(img_with_boxes, (wx, wy), 8, (0, 255, 0), -1)
                cv2.putText(img_with_boxes, "WRIST", (wx-20, wy-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw knife associations - LIGHT BLUE if wrist detected
    for assoc in knife_associations:
        if assoc.get("merged_box"):
            x1, y1, x2, y2 = assoc["merged_box"]
            color = (255, 255, 0) if assoc.get("wrist") else (255, 0, 255)
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_with_boxes, f"KNIFE HOLDER", (x1, y1-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if assoc.get("wrist"):
                wx, wy, _ = assoc["wrist"]
                cv2.circle(img_with_boxes, (wx, wy), 8, (255, 255, 0), -1)
                cv2.putText(img_with_boxes, "WRIST", (wx-20, wy-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return img_with_boxes

# ============================================
# FASTAPI APPLICATION
# ============================================

app = FastAPI(title="Gun & Knife Detection & Tracking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/test")
async def test():
    return {"message": "Backend is reachable"}

@app.get("/")
async def root():
    return {
        "message": "Gun & Knife Detection & Tracking API is running",
        "status": "active",
        "models_loaded": len(gun_models),
        "model_info": model_info,
        "directories": {
            "frames_dir": str(FRAMES_DIR),
            "output_dir": str(OUTPUT_DIR),
            "detected_images_dir": str(DETECTED_IMAGES_DIR)
        }
    }

@app.post("/detect-single")
async def detect_single(
    frame: UploadFile = File(...),
    session_id: str = Form(...),
    frame_index: int = Form(...)
):
    """Detect gun/knife in a single frame from webcam (1 frame per second) with tracking"""
    request_id = str(uuid.uuid4())[:8]
    print(f"\n{'='*50}")
    print(f"[Single {request_id}] Frame {frame_index} from session {session_id}")

    try:
        # Read and decode frame
        contents = await frame.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            print(f"[Single {request_id}] ERROR: Invalid frame data")
            return {
                "gun_detected": False,
                "knife_detected": False,
                "error": "Invalid frame data",
                "frame_index": frame_index
            }

        print(f"[Single {request_id}] Frame decoded: shape={img.shape}")

        # SAVE ALL INCOMING FRAMES to frames folder
        session_frames_dir = FRAMES_DIR / session_id
        session_frames_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        frame_filename = f"frame_{frame_index:06d}_{timestamp}.jpg"
        frame_filepath = session_frames_dir / frame_filename

        # Save original frame
        cv2.imwrite(str(frame_filepath), img)
        print(f"[Single {request_id}] ✓ Saved incoming frame: {frame_filename} (size: {frame_filepath.stat().st_size} bytes)")

        # Run detection
        start_time = time.time()

        # Detect humans and weapons
        humans = detect_humans(img, conf=0.4)
        guns, knives = detect_weapons(img, conf=0.6)

        # Find gun holders and knife holders
        gun_associations = find_gun_holder(img, humans, guns)
        knife_associations = find_knife_holder(img, humans, knives)

        # Extract new holders from both associations
        new_holders = []
        for assoc in gun_associations:
            if assoc.get("human_box"):
                new_holders.append(assoc["human_box"])
        for assoc in knife_associations:
            if assoc.get("human_box"):
                new_holders.append(assoc["human_box"])

        # Initialize tracking for this session if not exists
        if session_id not in tracked_holders_by_session:
            tracked_holders_by_session[session_id] = []

        # Update tracking
        tracked_holders = update_tracking(tracked_holders_by_session[session_id], humans)

        # Add new holders that aren't already tracked
        for nh in new_holders:
            already_tracked = False
            for th in tracked_holders:
                if iou(nh, th) > 0.5:
                    already_tracked = True
                    break
            if not already_tracked:
                tracked_holders.append(nh)

        # Store updated tracked holders
        tracked_holders_by_session[session_id] = tracked_holders

        detection_time = (time.time() - start_time) * 1000

        gun_detected = len(guns) > 0 or len(gun_associations) > 0
        knife_detected = len(knives) > 0 or len(knife_associations) > 0
        any_weapon_detected = gun_detected or knife_detected

        saved_image_path = None
        image_url = None

        print(f"[Single {request_id}] Detection result: {len(guns)} gun(s), {len(knives)} knife(s), {len(gun_associations)} gun associations, {len(knife_associations)} knife associations in {detection_time:.2f}ms")
        print(f"[Single {request_id}] Tracked holders: {len(tracked_holders)}")

        # If any weapon detected, save the annotated image to detected_images folder
        if any_weapon_detected:
            # Draw all annotations on the image
            img_with_boxes = draw_annotations(img, guns, knives, tracked_holders, gun_associations, knife_associations)

            # Generate unique filename for detected image
            weapon_type = "weapon"
            if gun_detected and knife_detected:
                weapon_type = "gun_and_knife"
            elif gun_detected:
                weapon_type = "gun"
            elif knife_detected:
                weapon_type = "knife"

            detected_filename = f"{weapon_type}_detected_{session_id}_{timestamp}_frame_{frame_index:06d}.jpg"
            detected_filepath = DETECTED_IMAGES_DIR / detected_filename

            # Save the annotated image
            cv2.imwrite(str(detected_filepath), img_with_boxes)
            saved_image_path = str(detected_filepath)
            image_url = f"/download-image/{detected_filename}"

            print(f"[Single {request_id}] ✓ WEAPON DETECTED! ({weapon_type}) Saved annotated image: {detected_filename}")

            # Store in session buffer
            if session_id not in recording_buffers:
                recording_buffers[session_id] = {
                    "images": [],
                    "frames": [],
                    "start_time": datetime.now().isoformat(),
                    "gun_detections": 0,
                    "knife_detections": 0,
                    "tracked_holders_count": 0
                }

            if gun_detected:
                recording_buffers[session_id]["gun_detections"] += 1
            if knife_detected:
                recording_buffers[session_id]["knife_detections"] += 1

            recording_buffers[session_id]["tracked_holders_count"] = len(tracked_holders)
            recording_buffers[session_id]["images"].append({
                "filename": detected_filename,
                "frame_index": frame_index,
                "timestamp": timestamp,
                "original_frame": frame_filename,
                "gun_count": len(guns),
                "knife_count": len(knives),
                "holder_count": len(tracked_holders),
                "weapon_type": weapon_type
            })
            recording_buffers[session_id]["frames"].append(frame_filename)

        result = {
            "gun_detected": gun_detected,
            "knife_detected": knife_detected,
            "any_weapon_detected": any_weapon_detected,
            "session_id": session_id,
            "frame_index": frame_index,
            "detection_time_ms": round(detection_time, 2),
            "gun_count": len(guns),
            "knife_count": len(knives),
            "holder_count": len(tracked_holders),
            "gun_association_count": len(gun_associations),
            "knife_association_count": len(knife_associations),
            "request_id": request_id,
            "saved_frame": frame_filename,
            "status": "success"
        }

        # Add image info if weapon was detected
        if any_weapon_detected and image_url:
            result["saved_image"] = image_url
            result["image_filename"] = detected_filename

            # Add detailed gun association info
            gun_associations_info = []
            for assoc in gun_associations:
                assoc_info = {
                    "weapon_type": "gun",
                    "has_wrist": assoc.get("wrist") is not None,
                    "weapon_box": assoc.get("weapon_box"),
                    "human_box": assoc.get("human_box")
                }
                if assoc.get("wrist"):
                    assoc_info["wrist_position"] = [assoc["wrist"][0], assoc["wrist"][1]]
                gun_associations_info.append(assoc_info)

            # Add detailed knife association info
            knife_associations_info = []
            for assoc in knife_associations:
                assoc_info = {
                    "weapon_type": "knife",
                    "has_wrist": assoc.get("wrist") is not None,
                    "weapon_box": assoc.get("weapon_box"),
                    "human_box": assoc.get("human_box")
                }
                if assoc.get("wrist"):
                    assoc_info["wrist_position"] = [assoc["wrist"][0], assoc["wrist"][1]]
                knife_associations_info.append(assoc_info)

            result["gun_associations"] = gun_associations_info
            result["knife_associations"] = knife_associations_info

        print(f"[Single {request_id}] Response: gun_detected={gun_detected}, knife_detected={knife_detected}, holders={len(tracked_holders)}")
        print(f"{'='*50}")
        return result

    except Exception as e:
        print(f"[Single {request_id}] ERROR: {e}")
        traceback.print_exc()
        return {
            "gun_detected": False,
            "knife_detected": False,
            "error": str(e),
            "frame_index": frame_index,
            "status": "error"
        }

@app.get("/list-frames")
async def list_frames(session_id: str = None):
    """List all saved frames"""
    frames = []
    try:
        if session_id:
            session_dir = FRAMES_DIR / session_id
            if session_dir.exists():
                for f in sorted(session_dir.glob("*.jpg")):
                    frames.append({
                        "filename": f.name,
                        "size": f.stat().st_size,
                        "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
                        "session_id": session_id
                    })
        else:
            for session_dir in FRAMES_DIR.iterdir():
                if session_dir.is_dir():
                    for f in sorted(session_dir.glob("*.jpg")):
                        frames.append({
                            "filename": f.name,
                            "size": f.stat().st_size,
                            "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
                            "session_id": session_dir.name
                        })
        frames.sort(key=lambda x: x['created'], reverse=True)
        return {"frames": frames, "count": len(frames)}
    except Exception as e:
        return {"frames": [], "count": 0, "error": str(e)}

@app.get("/list-detected-images")
async def list_detected_images():
    """List all detected weapon images"""
    images = []
    try:
        for f in DETECTED_IMAGES_DIR.glob("*.jpg"):
            # Determine weapon type from filename
            weapon_type = "unknown"
            if "gun" in f.name.lower():
                weapon_type = "gun"
            elif "knife" in f.name.lower():
                weapon_type = "knife"
            elif "gun_and_knife" in f.name.lower():
                weapon_type = "gun_and_knife"

            images.append({
                "filename": f.name,
                "size": f.stat().st_size,
                "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
                "url": f"/download-image/{f.name}",
                "weapon_type": weapon_type
            })
        images.sort(key=lambda x: x['created'], reverse=True)
        return {"images": images, "count": len(images)}
    except Exception as e:
        return {"images": [], "count": 0, "error": str(e)}

@app.get("/download-image/{filename}")
async def download_image(filename: str):
    """Download a detected weapon image"""
    file_path = DETECTED_IMAGES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path=str(file_path), media_type="image/jpeg", filename=filename)

@app.get("/list-videos")
async def list_videos():
    """List all processed videos"""
    videos = []
    try:
        for f in OUTPUT_DIR.glob("*.mp4"):
            videos.append({
                "filename": f.name,
                "size": f.stat().st_size,
                "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
                "url": f"/download/{f.name}"
            })
        videos.sort(key=lambda x: x['created'], reverse=True)
        return {"videos": videos, "count": len(videos)}
    except Exception as e:
        return {"videos": [], "count": 0, "error": str(e)}

@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download a processed video"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path=str(file_path), media_type="video/mp4", filename=filename)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    frames_count = sum(1 for _ in FRAMES_DIR.rglob("*.jpg"))
    detected_count = len(list(DETECTED_IMAGES_DIR.glob("*.jpg")))
    videos_count = len(list(OUTPUT_DIR.glob("*.mp4")))

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(gun_models),
        "model_info": model_info,
        "storage": {
            "frames_saved": frames_count,
            "detected_images": detected_count,
            "videos_processed": videos_count
        },
        "directories": {
            "frames": str(FRAMES_DIR),
            "detected": str(DETECTED_IMAGES_DIR),
            "outputs": str(OUTPUT_DIR)
        }
    }

@app.post("/end-session")
async def end_session(request: Request):
    """End a webcam session and clear tracking data"""
    try:
        data = await request.json()
        session_id = data.get('session_id')

        session_frames_dir = FRAMES_DIR / session_id
        frames_count = len(list(session_frames_dir.glob("*.jpg"))) if session_frames_dir.exists() else 0

        result = {
            "message": "Session ended",
            "session_id": session_id,
            "frames_saved": frames_count,
            "frames_directory": str(session_frames_dir) if session_frames_dir.exists() else None
        }

        # Clear tracking data for this session
        if session_id in tracked_holders_by_session:
            result["tracked_holders_cleared"] = len(tracked_holders_by_session[session_id])
            del tracked_holders_by_session[session_id]

        if session_id in recording_buffers:
            result["gun_detections"] = recording_buffers[session_id].get("gun_detections", 0)
            result["knife_detections"] = recording_buffers[session_id].get("knife_detections", 0)
            result["detected_images"] = len(recording_buffers[session_id]["images"])
            result["max_tracked_holders"] = recording_buffers[session_id]["tracked_holders_count"]
            del recording_buffers[session_id]

        print(f"Session {session_id} ended. Total frames saved: {frames_count}")
        return result

    except Exception as e:
        return {"error": str(e)}

@app.post("/process")
async def process_media(file: UploadFile = File(...)):
    """Process uploaded video file with full detection and tracking"""
    temp_input_path = None
    temp_output_path = None

    try:
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Only video files are supported")

        file_id = str(uuid.uuid4())
        temp_input_path = UPLOAD_DIR / f"{file_id}_input.mp4"
        temp_output_path = OUTPUT_DIR / f"{file_id}_output.mp4"

        content = await file.read()
        with open(temp_input_path, "wb") as f:
            f.write(content)

        print(f"Processing video: {file.filename}")
        print(f"Output will be saved to: {temp_output_path}")

        # Process video with detection and tracking
        cap = cv2.VideoCapture(str(temp_input_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(temp_output_path), fourcc, fps, (w, h))

        tracked_holders = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # Detect humans and weapons
            humans = detect_humans(frame, conf=0.4)
            guns, knives = detect_weapons(frame, conf=0.6)

            # Find associations
            gun_associations = find_gun_holder(frame, humans, guns)
            knife_associations = find_knife_holder(frame, humans, knives)

            # Extract new holders
            new_holders = []
            for assoc in gun_associations:
                if assoc.get("human_box"):
                    new_holders.append(assoc["human_box"])
            for assoc in knife_associations:
                if assoc.get("human_box"):
                    new_holders.append(assoc["human_box"])

            # Update tracking
            tracked_holders = update_tracking(tracked_holders, humans)

            # Add new holders
            for nh in new_holders:
                already_tracked = False
                for th in tracked_holders:
                    if iou(nh, th) > 0.5:
                        already_tracked = True
                        break
                if not already_tracked:
                    tracked_holders.append(nh)

            # Draw annotations
            annotated_frame = draw_annotations(frame, guns, knives, tracked_holders, gun_associations, knife_associations)
            out_video.write(annotated_frame)

            if frame_id % 30 == 0:
                print(f"Processed frame {frame_id}, tracked holders: {len(tracked_holders)}")

        cap.release()
        out_video.release()

        return FileResponse(
            path=str(temp_output_path),
            media_type="video/mp4",
            filename=f"detected_{file.filename}",
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if temp_input_path and temp_input_path.exists():
                os.remove(temp_input_path)
        except:
            pass

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check system status"""
    frames_count = sum(1 for _ in FRAMES_DIR.rglob("*.jpg"))
    sessions = [d.name for d in FRAMES_DIR.iterdir() if d.is_dir()]

    return {
        "frames_directory": str(FRAMES_DIR),
        "frames_directory_exists": FRAMES_DIR.exists(),
        "total_frames_saved": frames_count,
        "sessions": sessions,
        "session_count": len(sessions),
        "detected_images_count": len(list(DETECTED_IMAGES_DIR.glob("*.jpg"))),
        "videos_count": len(list(OUTPUT_DIR.glob("*.mp4"))),
        "active_sessions_with_tracking": len(tracked_holders_by_session),
        "tracking_data": {k: len(v) for k, v in tracked_holders_by_session.items()},
        "model_info": model_info
    }

# ============================================
# START THE API SERVER
# ============================================

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Start server in background
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

time.sleep(3)

print("\n" + "="*60)
print("✓ SERVER STARTED SUCCESSFULLY!")
print("="*60)
print(f"📍 Local URL: http://localhost:8000")
print(f"📁 Frames will be saved to: {FRAMES_DIR}")
print(f"📸 Detected images saved to: {DETECTED_IMAGES_DIR}")
print(f"🎬 Output videos saved to: {OUTPUT_DIR}")
print("="*60)

# ============================================
# SSH TUNNEL WITH SERVEO.NET
# ============================================

print("\n🌐 Setting up SSH tunnel with serveo.net...")

# Kill any existing SSH processes
!pkill ssh 2>/dev/null
time.sleep(2)

# Create SSH config
!mkdir -p ~/.ssh
!echo "Host serveo.net" > ~/.ssh/config
!echo "  StrictHostKeyChecking no" >> ~/.ssh/config
!echo "  UserKnownHostsFile /dev/null" >> ~/.ssh/config

# Start SSH tunnel
tunnel_process = subprocess.Popen(
    ['ssh', '-R', '80:localhost:8000', 'serveo.net'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

public_url = None
print("\nWaiting for tunnel URL...")
for line in iter(tunnel_process.stdout.readline, ''):
    print(line.strip())
    if 'https://' in line and '.serveo.net' in line:
        import re
        urls = re.findall(r'https://[a-z0-9-]+\.serveo\.net', line)
        if urls:
            public_url = urls[0]
            print(f"\n{'='*60}")
            print(f"✅ PUBLIC URL: {public_url}")
            print(f"{'='*60}")
            break

if public_url:
    print(f"""
{'='*60}
✅ API IS READY!
{'='*60}

🌐 PUBLIC URL: {public_url}

📁 FOLDER STRUCTURE:
   /content/frames/           ← ALL incoming frames (SAVED HERE)
   /content/detected_images/  ← ONLY weapon-detected images with annotations
   /content/outputs/          ← Processed videos

🔍 TEST ENDPOINTS:
   - Health: {public_url}/health
   - List Frames: {public_url}/list-frames
   - Debug: {public_url}/debug
   - Detected Images: {public_url}/list-detected-images

🎯 DETECTION FEATURES:
   - Gun detection (labels: guns, Gun, Pistol, handgun, rifle)
   - Knife detection (labels: knife)
   - Grenade detection (labels: Grenade)
   - Human detection and tracking
   - Wrist/hand pose estimation
   - Gun holder identification
   - Knife holder identification
   - Cross-frame tracking

📝 MODEL LABELS:
   - best.pt: {{0: 'guns', 1: 'knife'}}
   - best (3).pt: {{0: 'Gun'}}
   - best (2).pt: {{0: 'Grenade', 1: 'Gun', 2: 'Knife', 3: 'Pistol', 4: 'handgun', 5: 'rifle'}}
   - model.pt: {{0: 'Gun'}}
   - best (4).pt: {{0: 'Grenade', 1: 'Gun', 2: 'Knife', 3: 'Pistol', 4: 'handgun', 5: 'rifle'}}

📝 UPDATE YOUR REACT APP:
   Set Colab URL to: {public_url}

{'='*60}
""")

print("\n🔄 Server running... Press CTRL+C to stop\n")

# Keep alive
try:
    while True:
        time.sleep(60)
        # Print status every minute
        frames_count = sum(1 for _ in FRAMES_DIR.rglob("*.jpg"))
        if frames_count > 0:
            print(f"📊 Status: {frames_count} frames saved, {len(tracked_holders_by_session)} active tracking sessions")
except KeyboardInterrupt:
    print("\n\n✓ Server stopped")
    tunnel_process.terminate()