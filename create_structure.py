import os

# Define the folder structure
folder_structure = {
    "fusion_detection": {
        "config": ["yolo_pointnet_fusion_trainer.json"],
        "data": {
            "calibration": [],
            "left_images": [],
            "labels": [],
            "velodyne": []
        },
        "models": [
            "__init__.py",
            "pointnet_backbone.py",
            "yolo_backbone.py",
            "fusion_module.py"
        ],
        "utils": [
            "__init__.py",
            "dataset.py",
            "kitti_utils.py",
            "augmentation.py",
            "visualization.py"
        ],
        "": ["train.py", "eval.py", "README.md"]
    }
}

# Function to create folders and files
def create_structure(base_path, structure):
    for key, value in structure.items():
        folder_path = os.path.join(base_path, key)
        os.makedirs(folder_path, exist_ok=True)
        if isinstance(value, dict):
            create_structure(folder_path, value)
        elif isinstance(value, list):
            for file in value:
                open(os.path.join(folder_path, file), 'a').close()

# Create the folder structure
create_structure(".", folder_structure)