<img width="853" height="722" alt="KETQUA" src="https://github.com/user-attachments/assets/a1cb14e5-1326-4af2-8b81-aa81513a3d79" />
# LAB 2 - AIoT Data Preparation + Baseline Model + Deploy Demo

## 1. Project này dùng để làm gì?

Project này là bài mẫu cho sinh viên chạy trước khi phát triển theo nhóm.

Luồng bài mẫu:

```text
Public dataset / fallback sample
→ kiểm tra schema
→ làm sạch dữ liệu IoT
→ tạo feature dataset
→ chia train/test theo thời gian
→ train Logistic Regression baseline
→ tính anomaly_score bằng Z-score
→ sinh decision_log.csv
→ lưu model .joblib
→ deploy model bằng FastAPI
→ test API /predict
```

Dataset chính: **UCI Occupancy Detection**.  
Khi máy có Internet, script sẽ tải dữ liệu từ GitHub mirror của tác giả. Nếu lớp học không có Internet, project tự dùng file fallback cùng schema để sinh viên vẫn chạy được end-to-end.

## 2. Cấu trúc thư mục

```text
lab2_aiot_public_dataset_deploy/
├── data/
│   ├── DATA_SOURCES.md
│   └── occupancy_fallback_same_schema.csv      # được tạo nếu không tải được public dataset
├── notebooks/
│   └── 01_data_prep_baseline_deploy_ready.ipynb
├── src/
│   ├── data_utils.py                           # hàm tải data, clean, train, decision
│   ├── download_data.py                        # tải public dataset hoặc dùng fallback
│   ├── run_training_pipeline.py                # chạy pipeline không cần notebook
│   ├── app.py                                  # FastAPI deploy model
│   ├── test_api.py                             # test /health, /model-info, /predict
│   └── check_outputs.py                        # kiểm tra đã hoàn thành chưa
├── models/
│   └── occupancy_baseline.joblib               # model sinh ra sau khi chạy notebook
├── outputs/
│   ├── metrics.json
│   ├── decision_log.csv
│   └── figures/
└── requirements.txt
```

## 3. Cài môi trường

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Chạy bài mẫu bằng Jupyter Notebook

```bash
jupyter lab
```

Mở file:

```text
notebooks/01_data_prep_baseline_deploy_ready.ipynb
```

Chọn **Run → Run All Cells**.

> **Lưu ý quan trọng cho sinh viên:**
>
> - Chạy notebook **trước**, đọc và giải thích từng cell **sau**.  
>   Không bắt đầu bằng FastAPI vì bước deploy chỉ có ý nghĩa khi đã hiểu pipeline tạo ra model.
> - Sau khi chạy xong notebook, file `models/occupancy_baseline.joblib` được tạo ra — model đã được **đóng gói ra khỏi notebook**. FastAPI (`src/app.py`) chỉ load file này và phục vụ request, không cần chạy lại notebook mỗi lần.

Sau khi chạy xong, phải có các file:

```text
data/telemetry_clean.csv
data/feature_dataset.csv
models/occupancy_baseline.joblib
outputs/metrics.json
outputs/decision_log.csv
outputs/figures/01_co2_time_series.png
outputs/figures/02_confusion_matrix.png
outputs/figures/03_occupancy_probability.png
```

> **Về dataset và metric:**
>
> - Môi trường **không có Internet**: `src/download_data.py` tự dùng file `data/occupancy_fallback_same_schema.csv` — cùng schema, khác phân phối. Pipeline vẫn chạy end-to-end hoàn chỉnh.
> - Môi trường **có Internet**: script ưu tiên tải **UCI Occupancy Detection** từ GitHub mirror của tác giả → metric sẽ phản ánh dataset thực tế (~0.99 accuracy).
> - Metric có thể **khác nhau** giữa fallback và public dataset. Sinh viên **không cần đạt cùng con số tuyệt đối** — yêu cầu duy nhất là pipeline chạy đúng và giải thích được từng bước.

## 5. Chạy nhanh không cần notebook

```bash
python src/run_training_pipeline.py
python src/check_outputs.py
```

## 6. Deploy model bằng FastAPI

Mở terminal ở thư mục project, chạy:

```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

Mở trình duyệt:

```text
http://127.0.0.1:8000/docs
```

Test bằng terminal thứ hai:

```bash
python src/test_api.py
```

Kết quả đúng sẽ có dòng:

```text
API TEST PASSED: FastAPI model deployment is working.
```

## 7. Kiểm tra thế nào là hoàn thành?

Sinh viên hoàn thành bài mẫu khi:

1. Notebook chạy hết không lỗi.
2. Có model `models/occupancy_baseline.joblib`.
3. Có `outputs/metrics.json` với accuracy, precision, recall, f1.
4. Có `outputs/decision_log.csv` gồm `occupancy_probability`, `anomaly_score`, `is_anomaly`, `decision`, `command_hint`.
5. Chạy được FastAPI và truy cập được `/docs`.
6. Chạy `python src/test_api.py` thành công.
7. Giải thích được luồng: telemetry → features → model → decision → command hint.

## 8. Lưu ý

- Lab này chỉ deploy local để sinh viên hiểu model deployment cơ bản.
- Lab 5 sẽ phát triển inference service đầy đủ hơn: versioning, logging, validation, monitoring, API contract.
- Không cần lập trình ESP/MQTT ở Lab 2. Telemetry được lấy từ dataset công khai hoặc fallback sample.

---

## 9. Báo cáo kết quả thực nghiệm

### 9.1 Dataset và Schema

**Nguồn dữ liệu:** UCI Occupancy Detection Dataset (public, GitHub mirror)  
**Fallback:** `data/occupancy_fallback_same_schema.csv` (dùng khi offline)

| Trường | Kiểu | Mô tả |
|---|---|---|
| `date` | datetime | Timestamp đo đạc (1 phút/lần) |
| `Temperature` | float | Nhiệt độ phòng (°C) |
| `Humidity` | float | Độ ẩm tương đối (%) |
| `Light` | float | Cường độ ánh sáng (lux) |
| `CO2` | float | Nồng độ CO2 (ppm) |
| `HumidityRatio` | float | Tỷ lệ hơi nước (kg/kg không khí khô) |
| `Occupancy` | int | Nhãn: 1 = có người, 0 = không có người |

**Mapping về AIoT telemetry:** Các trường trên phản ánh đúng dữ liệu cảm biến môi trường phòng học thực tế trong hệ thống IoT (sensor node gửi qua MQTT → broker → lưu database → đưa vào pipeline AI).

---

### 9.2 Làm sạch dữ liệu IoT

**Kết quả làm sạch:**

| Tiêu chí | Kết quả |
|---|---|
| Tổng dòng ban đầu | 20,560 |
| Dòng sau làm sạch | 20,560 |
| Dòng bị loại | 0 |
| Timestamp lỗi | 0 |
| Duplicate rows | 0 |
| Outliers (mỗi cột) | 0 |
| Missing values | 0 |

**Giải thích:** Dataset UCI Occupancy là dataset chuẩn, đã được kiểm soát chất lượng. Tuy nhiên pipeline vẫn thực hiện đầy đủ các bước: kiểm tra timestamp, phát hiện duplicate, phát hiện outlier bằng Z-score (ngưỡng ±4σ), và forward-fill missing values — đảm bảo xử lý được dữ liệu thực tế từ sensor IoT.

---

### 9.3 Feature Dataset và Split

**Features đầu vào model:**

| Feature | Nguồn |
|---|---|
| `Temperature` | Raw sensor |
| `Humidity` | Raw sensor |
| `Light` | Raw sensor |
| `CO2` | Raw sensor |
| `HumidityRatio` | Raw sensor |
| `hour` | Trích xuất từ timestamp |
| `dayofweek` | Trích xuất từ timestamp |

**Train/test split theo thời gian** (tránh data leakage):
- Train: 75% dữ liệu đầu (chronological)
- Test: 25% dữ liệu cuối

---

### 9.4 AI Baseline Model

**Model:** Logistic Regression trong Pipeline (StandardScaler → LogisticRegression)  
**Version:** `lab2-v1`  
**Artifact:** `models/occupancy_baseline.joblib`

**Metrics trên tập test:**

| Metric | Giá trị |
|---|---|
| Accuracy | **0.9944** |
| Precision | **0.9751** |
| Recall | **0.9991** |
| F1 Score | **0.9869** |
| ROC AUC | **0.9988** |

**Confusion Matrix:**

```
                Predicted 0   Predicted 1
Actual 0           4015            28
Actual 1              1          1096
```

- False Negative (FN) = 1: Bỏ sót 1 trường hợp phòng có người → rủi ro thấp
- False Positive (FP) = 28: Báo có người khi phòng trống → lãng phí năng lượng nhỏ

---

### 9.5 Anomaly Score và Decision Log

**Anomaly score** tính bằng Z-score trên tập train (không dùng nhãn), đo mức độ bất thường của telemetry:

```
anomaly_score = mean(|z_i|) với z_i = (x_i - μ_train) / σ_train
```

**Decision rules:**

| Điều kiện | Decision | Command Hint |
|---|---|---|
| anomaly_score > 3.0 | `CHECK_SENSOR_OR_DATA` | `alert=SENSOR_CHECK` |
| CO2 > 1500 và occupancy_prob > 0.3 | `ALERT_VENTILATION_AND_TURN_FAN_ON` | `fan_state=ON; vent=FORCE_ON` |
| is_low_confidence (0.35–0.55) | `UNCERTAIN_HOLD_CURRENT_STATE` | `hold_state=True` |
| predicted_occupancy = 1 và Light < 50 | `ROOM_OCCUPIED_LIGHTING_NEEDED` | `light_state=ON` |
| predicted_occupancy = 1 | `ROOM_OCCUPIED_KEEP_COMFORT_MODE` | `ac_state=COMFORT` |
| predicted_occupancy = 0 | `ROOM_EMPTY_SAVE_ENERGY` | `ac_state=ECO; fan_state=OFF` |

**Decision log:** `outputs/decision_log.csv` — 200 dòng mẫu từ tập test, gồm đầy đủ: `occupancy_probability`, `anomaly_score`, `is_anomaly`, `is_low_confidence`, `confidence`, `decision`, `command_hint`, `safety_note`.

---

### 9.6 Deploy Model cơ bản

**FastAPI** chạy tại `http://127.0.0.1:8000`

| Endpoint | Method | Kết quả |
|---|---|---|
| `/health` | GET | `{"status":"ok","model_loaded":true,"model_version":"lab2-v1"}` |
| `/model-info` | GET | Tên model, features, metrics đầy đủ |
| `/predict` | POST | Trả về `occupancy_probability`, `anomaly_score`, `decision`, `command_hint`, `safety_note` |
| `/batch-predict` | POST | Xử lý nhiều bản ghi cùng lúc |
| `/docs` | GET | Swagger UI tương tác |

**Kết quả test:** `API TEST PASSED: FastAPI model deployment is working.`
<img width="1365" height="697" alt="Predict" src="https://github.com/user-attachments/assets/fc3949f3-2967-4755-a86b-9cda08ebda9a" />
<img width="1361" height="697" alt="Health" src="https://github.com/user-attachments/assets/76f60196-7ce2-4152-a532-53d343b5fd28" />

**Ví dụ response `/predict`:**
```json
{
  "model_output": {
    "occupancy_probability": 0.9969,
    "predicted_occupancy": 1,
    "anomaly_score": 2.2536,
    "is_anomaly": false,
    "is_low_confidence": false,
    "confidence": 0.4969
  },
  "decision": {
    "decision": "ROOM_OCCUPIED_KEEP_COMFORT_MODE",
    "command_hint": "ac_state=COMFORT",
    "safety_note": "Duy trì tiện nghi, không cần điều khiển mạnh."
  }
}
```

---
<img width="848" height="708" alt="Ketqua2" src="https://github.com/user-attachments/assets/e29ac04f-85f6-46b2-aa89-fe14dfa0709f" />
<img width="853" height="722" alt="ketqua" src="https://github.com/user-attachments/assets/aeacb6e4-83c1-4bf6-8494-605545f86b01" />
<img width="853" height="722" alt="ketqua" src="https://github.com/user-attachments/assets/ce7a4c15-64f8-4369-81e1-cdde163355c0" />
<img width="853" height="722" alt="KETQUA" src="https://github.com/user-attachments/assets/98633e9c-28fa-46b6-9cbe-27dbae53be1d" />
<img width="853" height="722" alt="KETQUA" src="https://github.com/user-attachments/assets/9cdfbe1a-3e5f-40f7-a1de-7fc3e0172c03" />

### 9.7 Giải thích kỹ thuật: Luồng Telemetry → Decision

```
Sensor IoT (ESP32/env_node)
    │
    ▼ telemetry JSON (Temperature, Humidity, Light, CO2, HumidityRatio, timestamp)
FastAPI /predict
    │
    ├─► Feature engineering (hour, dayofweek từ timestamp)
    ├─► StandardScaler (chuẩn hóa theo train_stats)
    ├─► LogisticRegression → occupancy_probability
    ├─► Z-score anomaly detection → anomaly_score
    │
    ▼
Decision engine
    ├─► Kiểm tra dữ liệu bất thường (sensor lỗi?)
    ├─► Kiểm tra CO2 nguy hiểm (cần thông gió?)
    ├─► Phân loại độ tin cậy (high/low confidence)
    └─► Ra quyết định + command_hint cho hệ thống actuator
```

---

### 9.8 Rủi ro dữ liệu sai và biện pháp

| Rủi ro | Mô tả | Biện pháp |
|---|---|---|
| Sensor drift | Cảm biến đọc lệch theo thời gian | Anomaly detection bằng Z-score; safety_note cảnh báo |
| Missing telemetry | Mất kết nối MQTT → null values | Forward-fill trong pipeline làm sạch |
| Duplicate timestamp | Gói tin gửi 2 lần | Drop duplicates trong clean step |
| CO2 spike giả | Nhiễu cảm biến | Ngưỡng anomaly_score > 3.0 → CHECK_SENSOR |
| Data leakage | Dùng future data để train | Train/test split theo thời gian (chronological) |
| Overfit metric | Accuracy = 1.0 trên fallback data | Dùng public UCI data → metrics thực tế ~0.99 |
| Low confidence zone | Model không chắc (prob 0.35–0.55) | Decision = UNCERTAIN_HOLD_CURRENT_STATE |
