# Save final augmented/original data and manifest for Batch 2 to Google Drive
import shutil
import os

# 1. Configuration for Batch 2
DRIVE_BASE_BATCH_2 = '/content/drive/MyDrive/Thesis_Data/SLRD_Dataset_raw_batch_2'
TARGET_FOLDER_NAME_BATCH_2 = 'origanl_data_and_Augmented_data_Batch2'
FINAL_DRIVE_DIR_BATCH_2 = os.path.join(DRIVE_BASE_BATCH_2, TARGET_FOLDER_NAME_BATCH_2)

# Local sources for Batch 2
AUG_DATA_DIR_BATCH_2 = '/content/Sindhi_VSR_Augmented_Dataset_Batch_2'
PADDED_DATA_DIR_BATCH_2 = '/content/Final_Padded_Data_Batch_2'
MANIFEST_FILE_BATCH_2 = '/content/Sindhi_VSR_Batch2_Augmented_Manifest.csv'

# Ensure the new Drive folder exists
os.makedirs(FINAL_DRIVE_DIR_BATCH_2, exist_ok=True)

def zip_and_upload_batch_2(source_dir, zip_name):
    if not os.path.exists(source_dir):
        print(f"⚠️ Source folder not found: {source_dir}")
        return

    local_zip = f"/content/{zip_name}.zip"
    print(f"📦 Zipping {zip_name}... (this may take a few minutes)")

    # Create the zip file
    shutil.make_archive(f"/content/{zip_name}", 'zip', source_dir)

    # Upload to Drive
    drive_dest = os.path.join(FINAL_DRIVE_DIR_BATCH_2, f"{zip_name}.zip")
    print(f"📤 Uploading {zip_name}.zip to Drive...")
    shutil.move(local_zip, drive_dest)
    print(f"✅ Successfully uploaded: {zip_name}.zip")

# 2. Execute Uploads for Batch 2
# Upload the Batch 2 Augmented Dataset
zip_and_upload_batch_2(AUG_DATA_DIR_BATCH_2, 'Sindhi_VSR_Augmented_Dataset_Batch_2')

# Upload the Batch 2 Original Padded Dataset
zip_and_upload_batch_2(PADDED_DATA_DIR_BATCH_2, 'Final_Padded_Data_Batch_2')

# 3. Upload the Batch 2 Manifest
if os.path.exists(MANIFEST_FILE_BATCH_2):
    shutil.copy(MANIFEST_FILE_BATCH_2, os.path.join(FINAL_DRIVE_DIR_BATCH_2, 'Sindhi_VSR_Batch2_Augmented_Manifest.csv'))
    print("✅ Successfully uploaded: Sindhi_VSR_Batch2_Augmented_Manifest.csv")

print(f"\n--- ALL BATCH 2 DATA SECURED IN: {FINAL_DRIVE_DIR_BATCH_2} ---")