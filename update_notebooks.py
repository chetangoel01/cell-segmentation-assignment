import json
import os

def update_nb(filepath):
    with open(filepath, 'r') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # Use two channels when loading
            source = source.replace("dapi, _ = load_fov_images", "dapi, polyt = load_fov_images")
            
            # Change nuclei model to cyto2
            source = source.replace("model_type=\"nuclei\"", "model_type=\"cyto2\"")
            
            # Change channels=[0, 0] to channels=[1, 2]
            source = source.replace("channels=[0, 0]", "channels=[1, 2]")
            
            # Add diameter=30 if not present in train_seg, wait train_seg doesn't take diameter directly.
            # In eval(): masks, _, _ = finetuned.eval(dapi[2], channels=[0, 0])
            source = source.replace("eval(dapi[2], channels=[0, 0])", "eval(np.stack([polyt[2], dapi[2]], axis=0), diameter=30, channels=[1, 2])")
            source = source.replace("eval(dapi[2], diameter=30, channels=[0, 0])", "eval(np.stack([polyt[2], dapi[2]], axis=0), diameter=30, channels=[1, 2])")
            source = source.replace("eval(image, diameter=30, channels=[0, 0])", "eval(image, diameter=30, channels=[1, 2])")
            
            # For appending to train_images
            source = source.replace("train_images.append(dapi[2])", "train_images.append(np.stack([polyt[2], dapi[2]], axis=0))")
            
            # For submission or evaluation running segmentation
            source = source.replace("masks = run_segmentation(dapi[2])", "masks = run_segmentation(np.stack([polyt[2], dapi[2]], axis=0))")
            
            # Split back into lines
            lines = source.splitlines(True)
            cell['source'] = lines
            
    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)

for nb_file in ["notebooks/03_finetune.ipynb", "notebooks/04_submission.ipynb", "pipeline.ipynb"]:
    if os.path.exists(nb_file):
        update_nb(nb_file)
        print(f"Updated {nb_file}")
