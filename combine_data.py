import os
import shutil
import random

def organize_breast_cancer_data(root_dir, output_dir="./data", train_ratio=0.8, seed=42):
    """
    Combines nested patient folders (e.g. Breast_Cancer/8863/0/, Breast_Cancer/8863/1/)
    into unified class folders 'non_idc/' and 'idc/', then splits 80/20 into Training and Testing.

    Args:
        root_dir (str): Path to the main Breast_Cancer directory (containing patient folders).
        output_dir (str): Directory to store the Training and Testing folders.
        train_ratio (float): Fraction of data for training.
        seed (int): Random seed for reproducibility.
    """

    random.seed(seed)

    # Target class folders
    class_map = {
        "0": "non_idc",
        "1": "idc"
    }

    combined_dir = os.path.join(output_dir, "Combined")
    os.makedirs(combined_dir, exist_ok=True)

    # Combine all images from all patient folders into two main folders
    print("🔄 Combining all patient folders...")
    for label, class_name in class_map.items():
        dest_dir = os.path.join(combined_dir, class_name)
        os.makedirs(dest_dir, exist_ok=True)

    for patient_folder in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        for label in class_map.keys():
            src_dir = os.path.join(patient_path, label)
            if not os.path.isdir(src_dir):
                continue

            dest_dir = os.path.join(combined_dir, class_map[label])
            for img_file in os.listdir(src_dir):
                src_path = os.path.join(src_dir, img_file)
                if os.path.isfile(src_path):
                    new_name = f"{patient_folder}_{label}_{img_file}"
                    shutil.copy2(src_path, os.path.join(dest_dir, new_name))

    # Now split each class folder into Training and Testing
    print("\n🔍 Splitting into Training and Testing sets...")
    train_dir = os.path.join(output_dir, "Training")
    test_dir = os.path.join(output_dir, "Testing")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in class_map.values():
        src_class_dir = os.path.join(combined_dir, class_name)
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        all_images = [f for f in os.listdir(src_class_dir) if os.path.isfile(os.path.join(src_class_dir, f))]
        random.shuffle(all_images)

        split_idx = int(len(all_images) * train_ratio)
        train_imgs = all_images[:split_idx]
        test_imgs = all_images[split_idx:]

        for img in train_imgs:
            shutil.copy2(os.path.join(src_class_dir, img), os.path.join(train_class_dir, img))
        for img in test_imgs:
            shutil.copy2(os.path.join(src_class_dir, img), os.path.join(test_class_dir, img))

        print(f"{class_name}: {len(train_imgs)} train, {len(test_imgs)} test")

    print(f"\n✅ Done. Training and Testing sets created at: {output_dir}")

# Example usage:
organize_breast_cancer_data("./data/Breast_Cancer_1", output_dir="./data/Breast_Cancer", train_ratio=0.8)
