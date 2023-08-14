import os
import json

dataset_root ='/home/data/yujin/Species/Species-60_MC'
output_dir = '/home/workspace/UniAD/data/Species-60'
class_json_output_dir = '/home/workspace/UniAD/data/Species-60/json_vis_decoder'

def create_json_file(data, json_file_path):
    with open(json_file_path, 'w') as json_file:
        for entry in data:
            json.dump(entry, json_file, separators=(',', ':'), ensure_ascii=False)
            json_file.write('\n')

class_folders = os.listdir(dataset_root)
test_data = []

for class_folder in class_folders:
    data = []
    class_path = os.path.join(dataset_root, class_folder)
    test_path = os.path.join(class_path, "test")

    for root, _, files in os.walk(test_path):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(root, filename)
                file_path = os.path.relpath(file_path, dataset_root)
                label_name = class_folder

                if root.endswith(os.path.join("test", "good")):
                    label = 0
                    json_entry = {
                        "filename": file_path,
                        "label": label,
                        "label_name": "good"
                    }
                    data.append(json_entry)
                else:
                    label = 1
                    clsname = class_folder
                    json_entry = {
                        "filename": file_path,
                        "label": label,
                        "label_name": "defective",
                        "clsname": clsname
                    }
                    data.append(json_entry)

    if class_folder != "test":
        test_data.extend(data)

for class_folder in class_folders:
    data = []
    class_path = os.path.join(dataset_root, class_folder)
    test_path = os.path.join(class_path, "test")

    for root, _, files in os.walk(test_path):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(root, filename)
                file_path = os.path.relpath(file_path, dataset_root)
                label_name = class_folder

                if root.endswith(os.path.join("test", "good")):
                    label = 0
                    json_entry = {
                        "filename": file_path,
                        "label": label,
                        "label_name": "good"
                    }
                    data.append(json_entry)
                else:
                    label = 1
                    clsname = class_folder
                    json_entry = {
                        "filename": file_path,
                        "label": label,
                        "label_name": "defective",
                        "clsname": clsname
                    }
                    data.append(json_entry)

    json_file_path = os.path.join(class_json_output_dir, f"test_{class_folder}.json")
    create_json_file(data, json_file_path)

test_json_file_path = os.path.join(output_dir, "test.json")
create_json_file(test_data, test_json_file_path)

train_data = []
for class_folder in class_folders:
    class_path = os.path.join(dataset_root, class_folder)
    train_good_path = os.path.join(class_path, "train", "good")

    for root, _, files in os.walk(train_good_path):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(root, filename)
                file_path = os.path.relpath(file_path, dataset_root)
                label_name = class_folder
                clsname = class_folder
                label = 0

                data_entry = {
                    "filename": file_path,
                    "label": label,
                    "label_name": label_name,
                    "clsname": clsname
                }
                train_data.append(data_entry)

train_json_file_path = os.path.join(output_dir, "train.json")
create_json_file(train_data, train_json_file_path)