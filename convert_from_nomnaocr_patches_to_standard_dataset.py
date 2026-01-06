from pathlib import Path
import shutil

def parse_line(line: str):
    """
    Expect: <path>\t<label>
    Fallback: split on whitespace into 2 parts if no tab found.
    """
    line = line.strip("\n\r")
    if not line.strip():
        return None

    if "\t" in line:
        parts = line.split("\t", 1)
    else:
        # Fallback for files that might be space-separated.
        parts = line.split(None, 1)

    if len(parts) != 2:
        return None

    img_rel, label = parts[0].strip(), parts[1].strip()
    if not img_rel or label is None:
        return None
    return img_rel, label

def prepare_dataset(
    train_txt: str,
    src_root: str,
    out_root: str = "dataset",
    out_ext: str = ".jpg",
    start_index: int = 1,
    digits: int = 5
):
    train_path = Path(train_txt)
    src_root = Path(src_root)
    out_root = Path(out_root)

    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels_path = out_root / "labels.txt"

    total = 0
    copied = 0
    skipped = 0

    labels_out = []

    with train_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parsed = parse_line(line)
            if parsed is None:
                # ignore empty or malformed lines (but keep a warning)
                if line.strip():
                    print(f"[WARN] Malformed line {line_no}: {line.strip()[:120]}")
                continue

            img_rel, label = parsed
            total += 1

            src_img = src_root / img_rel
            if not src_img.exists():
                print(f"[WARN] Missing image (line {line_no}): {src_img}")
                skipped += 1
                continue

            new_name = f"{start_index + copied:0{digits}d}{out_ext}"
            dst_img = images_dir / new_name

            try:
                shutil.copy2(src_img, dst_img)
            except Exception as e:
                print(f"[WARN] Failed to copy (line {line_no}): {src_img} -> {dst_img} | {e}")
                skipped += 1
                continue

            labels_out.append(label)
            copied += 1

    # Write labels only for successfully copied images (aligned by order)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8", newline="\n") as f:
        for lab in labels_out:
            f.write(lab + "\n")

    print("Done.")
    print(f"Lines parsed: {total}")
    print(f"Images copied: {copied}")
    print(f"Skipped: {skipped}")
    print(f"Images folder: {images_dir.resolve()}")
    print(f"Labels file:   {labels_path.resolve()}")

# Example usage:
# - train_txt: path to Train.txt
# - src_root:  the folder that contains the image folders shown in your screenshot ("Patches")
prepare_dataset(
    train_txt="Patches/Train.txt",
    src_root="Patches",       # change this if your dataset root folder has a different name/path
    out_root="dataset",
    out_ext=".jpg",           # keep everything as .jpg (matches your example)
    start_index=1,
    digits=5
)
