# In this script, we will process the videos from Batch 2 (S16 to S30) to extract the lip region of interest (ROI) and save them as .png images. After processing, we will zip the results and upload them to Google Drive. Finally, we will unzip the uploaded files and convert all .png images to .jpg format.

import os
print(len(os.listdir('/content')))

# followoing are the packages should be installed in the colab environment
# !pip install dlib opencv-python numpy
# !wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# !bunzip2 shape_predictor_68_face_landmarks.dat.bz2


import os
import cv2  # Fixed: changed from 'import cv' to 'import cv2'
import dlib
import numpy as np
import shutil
import zipfile



# 1. Setup Paths Modified for Batch 2
# Matches the unzipped folder path from your previous step
INPUT_DIR = '/content/local_dataset_batch_2/SLRD_Dataset_raw_batch_2'
LOCAL_OUTPUT_DIR = '/content/SLRD_Processed_Lips_Batch_2'

# Saves directly inside your Drive's batch 2 folder
DRIVE_FINISH_DIR = '/content/drive/MyDrive/Thesis_Data/SLRD_Dataset_raw_batch_2/Processed_Speakers_Batch_2'

# Ensure directories exist
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
os.makedirs(DRIVE_FINISH_DIR, exist_ok=True)

# 2. Initialize Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def extract_lips_roi(video_path, output_subfolder, target_size=(160, 80)):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            # Landmarks 48-67 represent the mouth area region
            points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])
            x, y, w, h = cv2.boundingRect(points)

            padding = 5
            lip_crop = gray[max(0, y-padding):y+h+padding, max(0, x-padding):x+w+padding]

            if lip_crop.size > 0:
                lip_resized = cv2.resize(lip_crop, target_size)
                lip_enhanced = cv2.equalizeHist(lip_resized)
                # Saved as .png format as requested
                cv2.imwrite(os.path.join(output_subfolder, f"frame_{frame_idx:03d}.png"), lip_enhanced)
        frame_idx += 1
    cap.release()

# 3. Main Loop with Speaker-by-Speaker Upload (S16 to S30)
speakers = sorted([d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))])

for speaker in speakers:
    zip_name = f"{speaker}_ROI.zip"
    drive_zip_path = os.path.join(DRIVE_FINISH_DIR, zip_name)

    # Check if this speaker is already processed and uploaded to Drive
    if os.path.exists(drive_zip_path):
        print(f"--- Skipping {speaker}: Already exists in Google Drive ---")
        continue

    print(f"--- Starting Processing for: {speaker} ---")
    speaker_input_path = os.path.join(INPUT_DIR, speaker)
    speaker_output_path = os.path.join(LOCAL_OUTPUT_DIR, speaker)
    os.makedirs(speaker_output_path, exist_ok=True)

    # Process every video for this specific speaker
    video_files = [f for f in os.listdir(speaker_input_path) if f.endswith('.mp4')]
    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        save_path = os.path.join(speaker_output_path, video_name)
        os.makedirs(save_path, exist_ok=True)

        extract_lips_roi(os.path.join(speaker_input_path, video_file), save_path)

    # 4. Zip and Upload immediately after speaker completion
    print(f"Zipping and Uploading {speaker} to Drive...")
    local_zip_path = f"/content/{zip_name}"

    # Zipping the speaker's folder specifically
    shutil.make_archive(local_zip_path.replace('.zip', ''), 'zip', speaker_output_path)

    # Move zip to Drive folder
    shutil.move(local_zip_path, drive_zip_path)

    # 5. Clean up local speaker folder to save Colab disk space
    shutil.rmtree(speaker_output_path)
    print(f"--- Successfully Completed and Uploaded: {speaker} ---")

print("All Batch 2 speakers processed and uploaded successfully!")






# Paths modee accorsing to the patha
# getting this data that I uploaded onto the google cloud (Batch 2)


# Paths modified for Batch 2
DRIVE_ROI_DIR = '/content/drive/MyDrive/Thesis_Data/SLRD_Dataset_raw_batch_2/Processed_Speakers_Batch_2'
LOCAL_ROI_BASE = '/content/Final_ROI_Unzipped_Batch_2'
RAW_DATASET_DIR = '/content/local_dataset_batch_2/SLRD_Dataset_raw_batch_2' # To read .txt labels for S16-S30

os.makedirs(LOCAL_ROI_BASE, exist_ok=True)

# Unzip all speaker ROI files for Batch 2 (S16 to S30)
for zip_file in sorted(os.listdir(DRIVE_ROI_DIR)):
    if zip_file.endswith('.zip'):
        print(f"Unzipping {zip_file}...")
        zip_path = os.path.join(DRIVE_ROI_DIR, zip_file)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract into a folder named after the speaker (e.g., S16)
                speaker_name = zip_file.split('_')[0]
                extract_path = os.path.join(LOCAL_ROI_BASE, speaker_name)

                zip_ref.extractall(extract_path)
        except Exception as e:
            print(f"Error unzipping {zip_file}: {e}")

print("Batch 2 unzipping complete.")



LOCAL_ROI_BASE_Batch_2 = '/content/Final_ROI_Unzipped_Batch_2'

print("Starting in-place PNG to JPG conversion...")

# Walk through all directories and subdirectories inside the folder
for root, dirs, files in os.walk(LOCAL_ROI_BASE_Batch_2):
    for file in files:
        if file.lower().endswith('.png'):
            png_path = os.path.join(root, file)

            # Read the PNG image
            img = cv2.imread(png_path)

            if img is not None:
                # Generate the new .jpg path
                jpg_path = os.path.splitext(png_path)[0] + '.jpg'

                # Save as JPG
                cv2.imwrite(jpg_path, img)

                # Delete the original PNG file
                os.remove(png_path)
            else:
                print(f"Warning: Could not read {png_path}")

print("Conversion complete! All images inside Final_ROI_Unzipped_Batch_2 are now .jpg")