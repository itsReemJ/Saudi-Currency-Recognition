import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm


def extract_features(image):
    sift = cv2.SIFT_create(nfeatures=5000)
    _, descriptors = sift.detectAndCompute(image, None)
    return descriptors


def prepare_reference_data(dataset_path):
    reference_data = {}
    folders = os.listdir(dataset_path)

    for folder_name in tqdm(folders, desc="Processing folders"):
        folder_path = os.path.join(dataset_path, folder_name)
        image_list = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (640, 480))
                desc = extract_features(img_resized)
                if desc is not None:
                    image_list.append({'desc': desc})
        if image_list:
            reference_data[folder_name] = image_list

    return reference_data


if __name__ == "__main__":
    dataset_path = r"Images Dataset"
    output_file = "descriptors.pkl"

    data = prepare_reference_data(dataset_path)
    with open(output_file, "wb") as f:
        pickle.dump(data, f)

    print(f"✅ Descriptors saved to {output_file}")

