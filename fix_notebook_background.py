import json
import os

def fix_nb(filepath):
    with open(filepath, 'r') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. Map 0 to "background" string in submission/validation cells
            source = source.replace('all_cluster_ids[spot_id] = int(cid)', 'all_cluster_ids[spot_id] = int(cid) if cid > 0 else "background"')
            
            # 2. Ensure diameter=30 is always used in eval
            # If eval(..., diameter=None, ...) or eval(...) without diameter, fix it.
            # (My previous script already added diameter=30 to many, but let's be thorough)
            if 'eval(' in source and 'diameter=' not in source:
                source = source.replace('eval(', 'eval(diameter=30, ')
            source = source.replace('diameter=None', 'diameter=30')
            
            # 3. Handle ARI calculation in fine-tuning notebook to match background types
            # if we compare pred_ids (int) to gt_ids (int), it's fine. 
            # But let's check if gt_mask is produced by boundaries_to_mask (which it is, so it's int).
            
            # Split back into lines
            lines = source.splitlines(True)
            cell['source'] = lines
            
    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)

for nb_file in ["notebooks/03_finetune.ipynb", "notebooks/04_submission.ipynb", "pipeline.ipynb"]:
    if os.path.exists(nb_file):
        fix_nb(nb_file)
        print(f"Fixed {nb_file}")
