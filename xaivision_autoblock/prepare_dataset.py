import os
import random
import shutil
import argparse

def prepare_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.3,
             test_ratio=0):

    images_dir = os.path.join(source_dir, "images")
    labels_dir = os.path.join(source_dir, "labels")
    # Make sure output directories exist
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Collect images
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    # Calculate split sizes
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Assign splits
    splits = {
        "train": images[:train_end],
        "valid": images[train_end:val_end],
        "test": images[val_end:]
    }

    # Copy files
    for split, split_images in splits.items():
        for img in split_images:
            img_src = os.path.join(images_dir, img)
            lbl_src = os.path.join(labels_dir, os.path.splitext(img)[0] + ".txt")
            
            img_dst = os.path.join(output_dir, "images", split, img)
            lbl_dst = os.path.join(
                output_dir, "labels", split, os.path.splitext(img)[0] + ".txt")
            
            shutil.copy(img_src, img_dst)
            if os.path.exists(lbl_src):  # some images might not have labels
                shutil.copy(lbl_src, lbl_dst)

    print(f"✅ Done! Train: {len(splits['train'])}, Valid: {len(splits['valid'])}, Test: {len(splits['test'])}")

def yolo_copy(source_dir, output_dir):
    print(f"Copying files only...")
    dirs = ('train', 'valid', 'test')
    for dir in dirs:
        source_images_dir = os.path.join(source_dir, f"{dir}/images")
        source_labels_dir = os.path.join(source_dir, f"{dir}/labels")
        dest_images_dir = os.path.join(output_dir, f"images/{dir}")
        dest_labels_dir = os.path.join(output_dir, f"labels/{dir}")
        images = os.listdir(source_images_dir)
        for image in images:
            img_src = os.path.join(source_images_dir, image)
            lbl_src = os.path.join(
                source_labels_dir, os.path.splitext(image)[0] + '.txt')
            os.makedirs(dest_images_dir, exist_ok=True)
            os.makedirs(dest_labels_dir, exist_ok=True)
            shutil.copy(img_src, dest_images_dir)
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, dest_labels_dir)
    print(f"✅ Done! files copied!")


def prepare_classes(source_dir):
    print(f"Preparing classes...")
    subtitute_classes = {
        0: 4, # airplane
    }

    # List all files in the source labels directory
    dirs = ('train', 'valid', 'test')
    for dir in dirs:
        labels_dir = os.path.join(source_dir, 'labels', dir)
        labels = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
        for label in labels:
            label_file = os.path.join(labels_dir, label)
            with open(label_file, 'r') as f:
                lines = f.readlines()

            fixed_lines = []
            for line in lines:
                cls, *_rest = line.strip().split()
                if int(cls) not in subtitute_classes.keys():
                    fixed_lines.append(line)
                    continue
                cls = subtitute_classes[int(cls)]
                fixed_lines.append(f'{cls} {" ".join(_rest)}')

            fixed_lines = '\n'.join(fixed_lines)
            with open(label_file, 'w') as f:
                f.write(fixed_lines)

    print(f'✅ Done! classes fixed.')


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--no-split', action='store_true')
    args.add_argument("--source_dir", type=str, required=True)
    args.add_argument("--output_dir", type=str, default='./dataset')
    args.add_argument("--train_ratio", type=float, default=0.8)
    args.add_argument("--val_ratio", type=float, default=0.2)
    args.add_argument("--test_ratio", type=float, default=0)
    args = args.parse_args()

    if args.no_split:
        yolo_copy(args.source_dir, args.output_dir)
    else:
        prepare_dataset(
            args.source_dir, args.output_dir,
            train_ratio=args.train_ratio, val_ratio=args.val_ratio,
            test_ratio=args.test_ratio)
    prepare_classes(args.output_dir)