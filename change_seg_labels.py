# remap_seg_labels_25_to_2.py
import os, glob

ROOT = "/home/student/project/output/yolo_data_seg"  # <- your dataset root

def collapse(cls_id: int) -> int:
    if 0 <= cls_id <= 14:  # NH*
        return 0
    if 15 <= cls_id <= 24:  # T*
        return 1
    raise ValueError(f"Unexpected class id {cls_id}")

def remap_dir(lab_dir: str):
    txts = glob.glob(os.path.join(lab_dir, "**", "*.txt"), recursive=True)
    changed = 0
    for p in txts:
        with open(p, "r") as f:
            lines = f.read().strip().splitlines()
        out = []
        for ln in lines:
            if not ln.strip(): 
                continue
            parts = ln.strip().split()
            cls = int(float(parts[0]))
            parts[0] = str(collapse(cls))
            out.append(" ".join(parts))
        with open(p, "w") as f:
            f.write("\n".join(out) + ("\n" if out else ""))
        changed += 1
    return changed

if __name__ == "__main__":
    n1 = remap_dir(os.path.join(ROOT, "labels", "train"))
    n2 = remap_dir(os.path.join(ROOT, "labels", "val"))
    print(f"✅ Remapped {n1+n2} label files to 2 classes (NH=0, T=1).")
    print(f"ℹ️  Now use YAML: {ROOT}/data.yaml with nc: 2, names: [NH, T]")