import kagglehub
import shutil
import os

def download_data():
    print("⬇️ Downloading dataset from Kaggle...")
    # Download latest version
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    print(f"✅ Dataset downloaded to: {path}")

    # Create data folder if it doesn't exist
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)

    print(f"📂 Copying files to '{data_folder}'...")
    # Copy all files from the downloaded path to the data folder
    for item in os.listdir(path):
        source = os.path.join(path, item)
        destination = os.path.join(data_folder, item)
        
        if os.path.isfile(source):
            shutil.copy2(source, destination)
            print(f"  - Copied: {item}")

    print(f"\n✨ All files saved to '{data_folder}' folder")

if __name__ == "__main__":
    download_data()
