# Kiến trúc Data Engine (2025)

## Phạm vi

Phiên bản này hiện thực hợp đồng BROKER-GPT-3 (v1.1, elaborated). Engine không tải dữ liệu thị trường từ API mà chỉ đọc các CSV đã có, áp dụng quy tắc hợp đồng và sinh bộ output chuẩn hoá + bundle cho ChatGPT.

## Luồng tổng quát

```
inputs (out/market/technical_snapshot.csv,
        out/presets/*.csv,
        data/portfolios/<profile>/portfolio.csv,
        out/portfolios/{<profile>_positions,<profile>_sector}.csv,
        data/portfolios/<profile>/order_history.csv,
        data/universe/vn100.csv,
        config/{params.yaml,blocklist.csv},
        out/news/news_score.csv?)
          │
          ▼
scripts/engine/data_engine.py
          │  (tuân thủ tick-size, lot, ngân sách, risk guard)
          ▼
outputs (out/market/trading_bands.csv,
         out/signals/{levels,sizing,signals}.csv,
         out/orders/<profile>_LO_latest.csv,
         out/orders/<profile>_LO_YYYYMMDD.csv*,
         out/run/<profile>_manifest.json,
         .artifacts/engine/<profile>_attachments_latest.zip)
```

(*chỉ tạo khi có lệnh)

## Thành phần chính

### EngineConfig

- `EngineConfig.from_yaml(path, profile)` đọc `config/data_engine.yaml` với cấu trúc tối giản:
  ```yaml
  paths:
    out: out
    data: data
    config: config
    bundle: .artifacts/engine
  ```
- Mọi đường dẫn được chuẩn hoá, bảo đảm không “thoát” repo. `bundle_dir` mặc định `.artifacts/engine/` (được git-ignore) và được tạo sẵn.
- Các property của config cung cấp đường dẫn cố định cho input/output theo hợp đồng và thay đổi theo `profile` (ví dụ `data/portfolios/<profile>/portfolio.csv`).

### DataEngine.run()

1. **Load & validate**
   - Fail-fast nếu thiếu `vn100.csv`, `technical_snapshot.csv`, `params.yaml` hoặc cột bắt buộc.
   - Tất cả ticker được chuẩn hoá về uppercase.
2. **Trading bands (`out/market/trading_bands.csv`)**
   - Tick HOSE: `<10 → 0.01`, `10–49.95 → 0.05`, `>=50 → 0.10`.
   - Trần = `floor_to_tick(Ref*1.07)`, sàn = `ceil_to_tick(Ref*0.93)`; nếu đảo ngược → swap và gắn guard `BAND_ERROR`.
3. **Levels (`out/signals/levels.csv`)**
   - Đọc từng preset CSV, hỗ trợ rule R1–R6 đúng như contract.
   - Giá near-touch/opportunistic được round-to-tick rồi clamp vào [Floor,Ceil].
   - `Limit_kVND` bo tròn 1000 VND gần nhất nhưng giữ nguyên tick hợp lệ.
4. **Sizing (`out/signals/sizing.csv`)**
   - `TargetQty = CurrentQty` (placeholder khi chưa có model) nên `DeltaQty` mặc định 0.
   - `MaxOrderQty = min(max_pct_adv*ADV20, budget_side/Last, max_qty_per_order)`.
   - `SliceCount` và `SliceQty` dựa trên `slice_adv_ratio` và chuẩn hoá về lot 100.
   - Tính thêm `LiquidityScore`, `VolatilityScore`, `TodayFilledQty/WAP` (từ fills nếu có).
5. **Signals (`out/signals/signals.csv`)**
   - Momentum/MeanRev heuristics đơn giản (RSI, Ret20d, MACD, Z-score, vị trí so với Floor).
   - `BandDistance = min((Ceil-Last)/ATR14, (Last-Floor)/ATR14)` (0 nếu ATR14=0).
   - Ghép `news_score.csv` (nếu có) và sector exposure để suy ra `SectorBias`.
   - `RiskGuards` gồm: `BLOCKLIST`, `ZERO_ATR`, `ZERO_LAST`, `LOW_LIQ`, `BAND_ERROR`, `UNKNOWN_RULE`, cộng thêm các guard phát sinh ở bước order.
6. **Orders (`out/orders/<profile>_LO_latest.csv`)**
   - Side = sign của `DeltaQty`. Nếu 0 → không có lệnh.
   - Chọn preset theo độ fit (momentum/mean-reversion/balanced) để lấy near-touch price.
   - Điều chỉnh aggressiveness (`low` giữ nguyên, `med` bớt 1 tick khi volatility cao hoặc news xấu, `high` tiến 1 tick nếu news tốt & band_distance >1).
   - Clamp về biên; thêm guard `CLAMPED` hoặc `NEAR_LIMIT` (khi chạm biên).
   - Kiểm soát ngân sách và lot size (>=100, <=500000, không vượt `MaxOrderQty`).
   - Nếu `LOW_LIQ` và khối lượng >100 → bỏ qua.
   - Ghi thêm snapshot ngày (`out/orders/<profile>_LO_YYYYMMDD.csv`) khi có lệnh.
7. **Manifest, quick check & bundle**
   - `out/run/<profile>_manifest.json` chứa `generated_at` (UTC), `source_files` (đường dẫn tương đối repo), `params_hash` (SHA-256 của `params.yaml`).
   - Trước khi nén, engine kiểm chứng từng output bắt buộc: đúng bộ cột, file tồn tại và trading_bands/levels/sizing/signals có ít nhất một dòng dữ liệu.
   - `<profile>_attachments_latest.zip` trong `.artifacts/engine/` đóng gói trading_bands, levels, sizing, signals, orders và manifest. Summary trả về `attachment_files` (tên trong zip) và `missing_attachments` (nếu file chưa tồn tại tại thời điểm zip).

### Risk guard propagation

- `base_flags` được dựng từ snapshot (ATR/ADV/Last = 0, blocklist, lỗi bands hoặc rule).
- Khi tạo lệnh, guard bổ sung (`CLAMPED`, `NEAR_LIMIT`) được hợp nhất và signals được cập nhật lại trước khi ghi file.

## Kiểm thử

`tests/test_data_engine.py` dựng sandbox repo tạm, tạo đầy đủ input giả (snapshot, presets, danh mục, fills, news, params, blocklist, universe) rồi chạy engine:
- Xác thực từng file output tồn tại, đúng cấu trúc và bundle chứa đầy đủ CSV.
- Kiểm tra manifest & bundle ghi đường dẫn tương đối chính xác.
- Đảm bảo file orders rỗng khi không có `DeltaQty` khác 0.
- Có test riêng xác nhận engine raise lỗi khi thiếu dữ liệu bắt buộc (ví dụ trading_bands rỗng).

## Lưu ý vận hành

- Engine không chỉnh sửa file input; mọi output nằm trong `out/`.
- Nếu cần tạo lệnh thật, bổ sung chiến lược đặt `TargetQty` trước khi chạy engine (ví dụ tạo file trung gian và thay đổi phần `_build_sizing`).
- Để attach cho ChatGPT, chỉ cần lấy `.artifacts/engine/<profile>_attachments_latest.zip` (đã gồm các CSV quan trọng).
- Không có network call trong engine → chạy deterministically, dễ kiểm soát kết quả.
- Workflow CI `portfolio-engine-attachments` chạy trên mọi push. Nếu commit chỉ ảnh hưởng file trong `data/portfolios/<profile>/` thì chỉ gọi `./broker.sh engine --profile <profile>` cho những profile bị đổi; ngược lại chạy tuần tự cho tất cả profile tìm thấy trong `data/portfolios/`. Artifact `.artifacts/engine/<profile>_attachments_latest.zip` được upload với retention 3 ngày.
- `data/universe/vn100.csv` được tạo lại mỗi lần gọi engine thông qua script `scripts.tools.build_universe` (đọc `data/industry_map.csv`).
