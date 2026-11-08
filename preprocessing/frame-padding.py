# This code applies frame padding using duplication technique and creates the Batch 2 manifest
import os
import shutil
import pandas as pd

TARGET_FRAMES = 49
dataset_mapping_batch_2 = []

# Paths explicitly configured for Batch 2 setup
LOCAL_ROI_BASE_Batch_2 = '/content/Final_ROI_Unzipped_Batch_2'
RAW_DATASET_DIR_Batch_2 = '/content/local_dataset_batch_2/SLRD_Dataset_raw_batch_2'

def perform_padding(folder_path):
    # Reads the newly converted .jpg files
    frames = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    num_original = len(frames)

    if num_original == 0:
        return

    # Duplicate frames if count is less than 49
    current_idx = num_original
    while current_idx < TARGET_FRAMES:
        frame_to_dup = frames[current_idx % num_original]
        src_path = os.path.join(folder_path, frame_to_dup)
        dst_path = os.path.join(folder_path, f"padded_{current_idx:03d}.jpg")
        shutil.copy(src_path, dst_path)
        current_idx += 1

# Iterate through speakers in the unzipped ROI folder (S16 to S30)
for speaker in sorted(os.listdir(LOCAL_ROI_BASE_Batch_2)):
    speaker_roi_path = os.path.join(LOCAL_ROI_BASE_Batch_2, speaker)
    if not os.path.isdir(speaker_roi_path):
        continue

    print(f"Processing labels and padding for: {speaker}")

    for video_folder in sorted(os.listdir(speaker_roi_path)):
        video_full_path = os.path.join(speaker_roi_path, video_folder)
        if not os.path.isdir(video_full_path):
            continue

        # 1. Apply Duplication Padding
        perform_padding(video_full_path)

        # 2. Get Sindhi Label from the original .txt file matching Batch 2 structure
        txt_filename = video_folder + ".txt"
        txt_path = os.path.join(RAW_DATASET_DIR_Batch_2, speaker, txt_filename)

        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                sindhi_word = f.read().strip()

            dataset_mapping_batch_2.append({
                'video_id': video_folder,
                'speaker': speaker,
                'sindhi_label': sindhi_word,
                'path': video_full_path
            })
        else:
            print(f"Warning: Missing label file at {txt_path}")

# Save the final mapping to a specific Batch 2 CSV for training alignment
df_batch_2 = pd.DataFrame(dataset_mapping_batch_2)
manifest_output_path = '/content/Sindhi_LipReading_Batch2_Manifest.csv'
df_batch_2.to_csv(manifest_output_path, index=False, encoding='utf-8-sig')

print(f"\nProcessing Complete! Batch 2 Manifest created at: {manifest_output_path}")