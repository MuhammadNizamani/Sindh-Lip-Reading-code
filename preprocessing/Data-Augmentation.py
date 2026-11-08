# Data Augmentation Setup for Batch 2
import os
import pandas as pd
import cv2
import numpy as np

# 1. Setup Correct Paths for Batch 2
LOCAL_EXTRACT_DIR_BATCH_2 = '/content/Final_Padded_Data_Batch_2'
MANIFEST_PATH_BATCH_2 = '/content/Sindhi_LipReading_Batch2_Manifest.csv'
AUGMENTED_OUTPUT_DIR_BATCH_2 = '/content/Sindhi_VSR_Augmented_Dataset_Batch_2'
os.makedirs(AUGMENTED_OUTPUT_DIR_BATCH_2, exist_ok=True)

# 2. Load and Fix Manifest Paths for Batch 2
df_manifest_batch_2 = pd.read_csv(MANIFEST_PATH_BATCH_2)

def sync_path_batch_2(row):
    # Builds the direct path matching the local extraction layout
    return os.path.join(LOCAL_EXTRACT_DIR_BATCH_2, row['speaker'], row['video_id'])

df_manifest_batch_2['folder_path'] = df_manifest_batch_2.apply(sync_path_batch_2, axis=1)

# 3. Augmentation Functions
def apply_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def apply_translation(img):
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, 5], [0, 1, 5]])
    return cv2.warpAffine(img, M, (cols, rows))

def apply_jitter(img):
    return cv2.convertScaleAbs(img, alpha=1.2, beta=20)

def apply_rotate_neg10(img):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -10, 1)
    return cv2.warpAffine(img, M, (cols, rows))

augmentations = {
    'blur': apply_blur,
    'translate': apply_translation,
    'jitter': apply_jitter,
    'rotate_n10': apply_rotate_neg10
}

new_entries_batch_2 = []
print("Starting Augmentation on Sindhi ROIs (Batch 2)...")

for index, row in df_manifest_batch_2.iterrows():
    original_folder = row['folder_path']
    if not os.path.exists(original_folder):
        continue

    for aug_name, aug_func in augmentations.items():
        new_video_id = f"{row['video_id']}_{aug_name}"
        # Organized by speaker to match original dataset structure (S16 to S30)
        new_folder_path = os.path.join(AUGMENTED_OUTPUT_DIR_BATCH_2, row['speaker'], new_video_id)
        os.makedirs(new_folder_path, exist_ok=True)

        # Reads the newly padded .jpg files
        frames = sorted([f for f in os.listdir(original_folder) if f.endswith('.jpg')])
        for frame_file in frames:
            img = cv2.imread(os.path.join(original_folder, frame_file))
            if img is not None:
                # Apply transformation and save
                transformed_img = aug_func(img)
                cv2.imwrite(os.path.join(new_folder_path, frame_file), transformed_img)

        new_entries_batch_2.append({
            'video_id': new_video_id,
            'speaker': row['speaker'],
            'sindhi_label': row['sindhi_label'],
            'folder_path': new_folder_path,
            'frame_count': 49
        })

# 4. Save Final Manifest specifically for Batch 2
df_final_batch_2 = pd.concat([df_manifest_batch_2, pd.DataFrame(new_entries_batch_2)], ignore_index=True)
output_manifest_path = '/content/Sindhi_VSR_Batch2_Augmented_Manifest.csv'
df_final_batch_2.to_csv(output_manifest_path, index=False, encoding='utf-8-sig')

print(f"\nSuccess! Total labeled sequences in Batch 2 Manifest: {len(df_final_batch_2)}")
print(f"Manifest saved to: {output_manifest_path}")