# SISR-GAN: Super-Resolution Using GANs

SISR-GAN là một dự án sử dụng Generative Adversarial Networks (GANs) để tăng cường độ phân giải của hình ảnh.

## 📂 Cấu trúc thư mục
```
SISR-GAN/
│── data/                     # Thư mục chứa dữ liệu ảnh
│   ├── train/                # Ảnh huấn luyện
│   ├── test/                 # Ảnh kiểm tra
|── notebooks/                # Chứa các notebooks thử nghiệm
│   ├── generator.ipynb
│   ├── discriminator.ipynb
│── models/                   # Chứa các mô hình GAN
│   ├── generator.py          # Mô hình Generator
│   ├── discriminator.py      # Mô hình Discriminator
│── utils/                    # Các hàm tiện ích
│   ├── dataset.py            # Hàm xử lý dữ liệu
│   ├── loss.py               # Hàm tính loss
│── checkpoints/              # Lưu checkpoint của model
│── results/                  # Chứa ảnh đầu ra từ mô hình
│── train.py                  # Script huấn luyện GAN
│── test.py                   # Script kiểm tra model
│── requirements.txt          # Danh sách thư viện cần thiết
│── README.md                 # Hướng dẫn sử dụng project
```

## 🚀 Cài đặt
Yêu cầu Python 3.8+ và các thư viện liên quan:
```bash
pip install -r requirements.txt
```

## 🔥 Huấn luyện mô hình
Chạy lệnh sau để huấn luyện GAN:
```bash
python train.py --epochs 50 --batch_size 16
```

## 📊 Kiểm tra mô hình
Chạy mô hình trên ảnh kiểm tra:
```bash
python test.py --input data/test/sample.jpg --output results/sample_sr.jpg
```

## 📌 Ghi chú
- Hãy đảm bảo bạn có dữ liệu ảnh trước khi chạy huấn luyện.
- Bạn có thể tùy chỉnh tham số trong `train.py` và `test.py`.

## 📜 License
MIT License.

