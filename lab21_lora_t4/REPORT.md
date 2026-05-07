# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Văn A — 12345678
**Ngày nộp**: 2026-05-07
**Submission option**: A (lightweight)

## 1. Setup
- **Base model**: unsloth/Qwen2.5-3B-bnb-4bit
- **max_seq_length**: 256
- **GPU**: Tesla T4

## 2. Rank Experiment Results

|   rank |   alpha |   trainable_params |   train_time_min |   peak_vram_gb |   eval_loss |   eval_perplexity |
|-------:|--------:|-------------------:|-----------------:|---------------:|------------:|------------------:|
|      8 |      16 |        1.8432e+06  |          3.28892 |        11.0066 |     1.55982 |           4.75797 |
|     16 |      32 |        3.6864e+06  |          3.35279 |        10.4063 |     1.51349 |           4.54254 |
|     64 |     128 |        1.47456e+07 |          3.31757 |        11.7858 |     1.47582 |           4.37464 |

## 3. Conclusion về Rank Trade-off
- **Rank 16** là điểm cân bằng (Sweet spot) tốt nhất giữa chất lượng và tài nguyên.
- **Rank 64** cho thấy dấu hiệu hiệu suất giảm dần (diminishing returns) trên tập dữ liệu nhỏ này.
