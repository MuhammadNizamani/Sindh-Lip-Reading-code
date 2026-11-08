# Fixed code to extract labels and create the manifest for Batch 2
import os
import pandas as pd

# Paths updated strictly for Batch 2 structures
LOCAL_ROI_BASE_BATCH_2 = '/content/Final_ROI_Unzipped_Batch_2'
RAW_LABELS_DIR_BATCH_2 = '/content/local_dataset_batch_2/SLRD_Dataset_raw_batch_2'
MANIFEST_PATH_BATCH_2 = '/content/Sindhi_LipReading_Batch2_Manifest.csv'

dataset_mapping_batch_2 = []

# List all speaker folders in your processed ROI directory (S16 to S30)
speakers = [d for d in os.listdir(LOCAL_ROI_BASE_BATCH_2) if os.path.isdir(os.path.join(LOCAL_ROI_BASE_BATCH_2, d))]

print(f"Found {len(speakers)} speaker folders in Batch 2 ROI directory.")

for speaker in sorted(speakers):
    speaker_roi_path = os.path.join(LOCAL_ROI_BASE_BATCH_2, speaker)

    # List all video folders for this speaker (e.g., W07_S27_01)
    video_folders = [v for v in os.listdir(speaker_roi_path) if os.path.isdir(os.path.join(speaker_roi_path, v))]

    print(f"Processing {len(video_folders)} videos for {speaker}...")

    for video_folder in sorted(video_folders):
        video_full_path = os.path.join(speaker_roi_path, video_folder)

        # Construct the label path for Batch 2 mapping structure
        txt_filename = f"{video_folder}.txt"
        label_path = os.path.join(RAW_LABELS_DIR_BATCH_2, speaker, txt_filename)

        if os.path.exists(label_path):
            try:
                # Read the Sindhi word from the .txt file
                with open(label_path, 'r', encoding='utf-8') as f:
                    sindhi_word = f.read().strip()

                if sindhi_word:
                    dataset_mapping_batch_2.append({
                        'video_id': video_folder,
                        'speaker': speaker,
                        'sindhi_label': sindhi_word,
                        'folder_path': video_full_path,
                        'frame_count': len([f for f in os.listdir(video_full_path) if f.endswith('.jpg')])
                    })
            except Exception as e:
                print(f"Error reading {label_path}: {e}")
        else:
            # Debugging print if file is missing
            print(f"Warning: Label file not found at {label_path}")

# Create the DataFrame
df_batch_2 = pd.DataFrame(dataset_mapping_batch_2)

if not df_batch_2.empty:
    # Save with utf-8-sig to ensure Sindhi characters display correctly
    df_batch_2.to_csv(MANIFEST_PATH_BATCH_2, index=False, encoding='utf-8-sig')
    print(f"\nSuccess! Batch 2 Manifest saved to {MANIFEST_PATH_BATCH_2}")
    print(f"Total labeled sequences: {len(df_batch_2)}")
    print("\nPreview of first 5 rows:")
    print(df_batch_2[['video_id', 'sindhi_label']].head())
else:
    print("\nError: The manifest is still empty. Please check if your folder structure under RAW_LABELS_DIR matches the speaker IDs.")