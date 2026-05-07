# Báo cáo Lab 21: LoRA/QLoRA Fine-tuning Qwen2.5-3B

**Học viên:** Đào Quang Thắng  
**Mã HV:** 2A202600030  
**Ngày thực hiện:** 07/05/2026  
**Môi trường:** Google Colab (Tesla T4 - 16GB VRAM)

---

## 1. Khám phá Dataset
Chúng ta sử dụng dataset `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` được dịch từ Alpaca GPT-4 sang tiếng Việt.
- **Số lượng mẫu:** 200 mẫu được shuffle và chọn lọc.
- **Định dạng:** Alpaca format (Instruction, Input, Output).
- **Phân phối độ dài Token:** 
![Token Length](token_length.png)
Hầu hết các câu có độ dài dưới 150 tokens, phù hợp với giới hạn `max_seq_length=256` để tối ưu bộ nhớ.

---

## 2. Cấu hình Huấn luyện
Mô hình nền tảng là **Qwen2.5-3B** được load ở định dạng 4-bit (NF4) để tiết kiệm VRAM.
- **Phương pháp:** QLoRA (Quantized Low-Rank Adaptation).
- **Target Modules:** `q_proj`, `v_proj` (theo yêu cầu Lab).
- **Hyperparameters:**
  - `epochs`: 3
  - `batch size`: 1
  - `gradient accumulation`: 8 (Effective batch size = 8)
  - `learning rate`: 2e-4
  - `weight decay`: 0.01

---

## 3. Rank Experiment (r=8, 16, 64)
Chúng ta tiến hành thực nghiệm trên 3 mức rank khác nhau để đánh giá sự đánh đổi giữa hiệu năng và tài nguyên.

| Rank | LoRA Alpha | Trainable Params | Time (min) | Peak VRAM (GB) | Eval Loss | Perplexity |
|------|------------|------------------|------------|----------------|-----------|------------|
| 8    | 16         | 1,843,200        | 3.29       | 11.01          | 1.5598    | 4.76       |
| 16   | 32         | 3,686,400        | 3.35       | 10.41          | 1.5135    | 4.54       |
| 64   | 128        | 14,745,600       | 3.32       | 11.79          | 1.4758    | 4.37       |

---

## 4. Phân tích Loss Curve
Đồ thị dưới đây cho thấy quá trình hội tụ của mô hình trong 3 epoch huấn luyện.
![Loss Curve](loss_curve.png)
**Nhận xét:**
- Loss giảm đều qua các epoch, cho thấy mô hình đang học tốt từ dataset tiếng Việt.
- Rank càng lớn (r=64), loss cuối cùng càng thấp, chứng tỏ khả năng học phức tạp hơn. Tuy nhiên, sự chênh lệch giữa r=16 và r=64 là không quá lớn so với sự gia tăng số lượng tham số (gấp 4 lần).

---

## 5. Kết quả Qualitative (Trước/Sau)
So sánh câu trả lời của mô hình gốc (Base) và mô hình sau khi Fine-tune (Finetuned - r=16).

| Prompt | Base Model | Finetuned Model |
|--------|------------|-----------------|
| **Giải thích machine learning** | "Machine learning là một phân khúc của AI..." (Bị cắt câu nửa chừng) | "Machine learning là một bộ môn công nghệ máy tính... học từ dữ liệu và đưa ra dự đoán." (Hoàn chỉnh hơn) |
| **Code Fibonacci (Python)** | (Đoạn code bị lỗi logic hoặc cắt ngang) | (Đoạn code sử dụng vòng lặp tối ưu, logic sạch sẽ) |
| **Phân biệt RAG và Fine-tuning** | (Trả lời chung chung, thiếu sự phân định rõ ràng) | (Phân biệt rõ: Prompt Engineering là gợi ý, RAG là truy xuất, Fine-tuning là dạy lại kiến thức) |

---

## 6. Kết luận & Phân tích chi phí
### Kết luận về Rank
- **Rank 16** là điểm cân bằng (Sweet spot) tốt nhất cho tác vụ này trên T4. Nó mang lại Perplexity (4.54) thấp hơn Rank 8 (4.76) đáng kể trong khi chỉ tăng nhẹ thời gian huấn luyện.
- **Rank 64** mang lại độ chính xác cao nhất (PPL 4.37) nhưng tiêu tốn nhiều VRAM hơn và có nguy cơ Overfitting nếu dataset quá nhỏ.

### Phân tích chi phí
- Với thời gian huấn luyện tổng cộng khoảng 10-15 phút cho 3 thí nghiệm trên Colab T4:
  - Nếu dùng Google Colab Pro (khoảng $0.1/giờ), chi phí chưa tới **$0.02**.
  - Đây là giải pháp cực kỳ tiết kiệm để triển khai mô hình tiếng Việt chất lượng cao.

---
**Học viên cam kết nội dung báo cáo trung thực và tự thực hiện.**
