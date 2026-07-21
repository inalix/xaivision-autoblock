# XAI Vision AutoBlock


## Installation

### Using UV
1. Clone the repository
```bash
git clone <repository-url>
cd xaivision-autoblock
```
2. Buat virtual environment
```bash
uv venv
```
3. Aktifkan virtual environment
```bash
source .venv/bin/activate
```
4. Install dependencies
```bash
uv sync
```

## Download Raw Dataset & Prepare Dataset
```bash
# DOWNLOAD
git clone https://github.com/inalix/xaivision-raw-dataset
# Semua dataset dari kita sendiri akan ada di folder xaivision-raw-dataset/raw_dataset_autoblock/inalix/
# Dataset dari robolow akan ada di folder xaivision-raw-dataset/raw_dataset_autoblock/other/

# PREPARE DATASET
# from ROBOFLOW
python xaivision_autoblock/prepare_dataset.py --source_dir <path_to_raw_dataset> --output_dir <path_to_output_dataset default: dataset> --no-split
# ex
python xaivision_autoblock/prepare_dataset.py --source_dir /home/inalix/projects/xaivision-raw-dataset/raw_dataset_autoblock/other/airplane.v5i.yolo26 --output_dir ./dataset --no-split

# from INALIX (Label Studio export)
python xaivision_autoblock/prepare_dataset.py --source_dir <path_to_raw_dataset> --output_dir <path_to_output_dataset default: dataset>
# ex
python xaivision_autoblock/prepare_dataset.py --source_dir /home/inalix/projects/xaivision-raw-dataset/raw_dataset_autoblock/inalix/dataset1 --output_dir ./dataset
```

## Train Model
```bash
# BUKA data-train.yml atau copy data-train.yml dan pastikan path ke dataset sudah benar
# jalankan perintah ini:
yolo detect train model=yolo26s.pt data=data-train.yml epochs=300 cache=ram batch=0.90
# model = 26n, 26s, 26m, 26l, 26x
# data = path to data-train.yml
# epochs = number of epochs
```

## Jalankan program
### prepare environment
```bash
cp .env.example .env
vi .env # pastikan semua sesuai

# RUN DEVELOPMENT STREAM
python main.py
```