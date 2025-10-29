# Broker GPT Data Engine

> Công cụ này không đưa ra lời khuyên đầu tư. Nó đọc các snapshot CSV đã có, áp dụng hợp đồng dữ liệu BROKER-GPT-3 và xuất ra những file đầu ra chuẩn hoá để bạn xem xét thủ công.

## Engine làm gì?

Mỗi lần chạy, engine sẽ:

1. Đọc `out/market/technical_snapshot.csv` (giá/indicator), toàn bộ preset trong `out/presets/*.csv`, danh mục hiện tại (`data/portfolios/alpha.csv`), thông tin PnL/sector (`out/portfolios/*.csv`), lịch sử fills (`data/order_history/alpha_fills.csv`), budget trong `config/params.yaml`, blocklist và universe `data/universe/vn100.csv`.
2. Tính toán các bảng theo đúng đặc tả v1.1:
   - `out/market/trading_bands.csv`
   - `out/signals/levels.csv`
   - `out/signals/sizing.csv`
   - `out/signals/signals.csv`
   - `out/orders/alpha_LO_latest.csv` + bản snapshot ngày (`out/orders/alpha_LO_YYYYMMDD.csv` khi có lệnh)
   - `out/run/manifest.json` (liệt kê input, hash params)
3. Gom toàn bộ output chính vào `.artifacts/engine/attachments_latest.zip`, đồng thời trả về danh sách file thu thập được và file thiếu trong tóm tắt.

Trước khi nén, engine chạy quick check đảm bảo mỗi file theo hợp đồng đều tồn tại, đúng cấu trúc cột và (với các bảng bắt buộc như trading_bands/levels/sizing/signals) có ít nhất một dòng dữ liệu. Nếu thiếu, engine dừng với lỗi rõ ràng.

Mọi tính toán tuân thủ tick-size HOSE, giới hạn lot, kiểm tra ngân sách và bộ guard theo hợp đồng.

## Chuẩn bị

- Python 3.10+
- `pip install -r requirements.txt`
- Cập nhật các CSV đầu vào trước khi chạy engine (snapshot thị trường, presets, danh mục, fills, v.v.)

## Cấu hình

`config/data_engine.yaml` định nghĩa đường dẫn gốc (tương đối repo):

```yaml
paths:
  out: out
  data: data
  config: config
  bundle: .artifacts/engine
```

`config/params.yaml` phải có các khóa:

```yaml
buy_budget_vnd: 0
sell_budget_vnd: 0
max_order_pct_adv: 0.05
aggressiveness: low   # low | med | high
slice_adv_ratio: 0.02
min_lot: 100
max_qty_per_order: 500000
```

`config/blocklist.csv` (Ticker,Reason) và `data/universe/vn100.csv` (Ticker) là bắt buộc.

## Chạy engine

```bash
./broker.sh engine
```

Hoặc trực tiếp:

```bash
python -m scripts.engine.data_engine --config config/data_engine.yaml
```

Output tóm tắt (STDOUT) là JSON chứa số lượng mã, số lệnh, đường dẫn bundle và các file đính kèm.

## Các file đầu ra chính

| File | Nội dung | Inputs | Ghi chú |
| ---- | -------- | ------ | ------- |
| `out/market/trading_bands.csv` | TickSize, trần/sàn HOSE cho từng mã | `technical_snapshot.csv` | Tự động đổi chỗ khi sàn > trần và gắn `BAND_ERROR` vào guard |
| `out/signals/levels.csv` | Giá near-touch/opportunistic theo preset | Snapshot + bands + presets | Tôn trọng tick & biên, Limit_kVND bo tròn gần nhất |
| `out/signals/sizing.csv` | Target/Delta/MaxOrder, slice | Snapshot + danh mục + fills + params | Nếu chưa có chiến lược, Target = Current |
| `out/signals/signals.csv` | Momentum/MeanRev fit, BandDistance, News, SectorBias, RiskGuards | Snapshot + bands + sector + news + blocklist | RiskGuards gồm BLOCKLIST/ZERO_ATR/ZERO_LAST/LOW_LIQ/... |
| `out/orders/alpha_LO_latest.csv` | Lệnh giới hạn (kVND) | Levels + sizing + signals + params | Tự động skip BLOCKLIST/LOW_LIQ, clamp biên và thêm guard |
| `out/run/manifest.json` | `generated_at`, `source_files`, `params_hash` | -- | Liệt kê tất cả file input thực tế |

Nếu không có DeltaQty khác 0, file lệnh vẫn tồn tại nhưng rỗng.

## Kiểm thử

```bash
./broker.sh tests
```

Test chính `tests/test_data_engine.py` dựng dữ liệu giả theo hợp đồng rồi kiểm tra toàn bộ file output, manifest và bundle.

## GitHub Actions

- Workflow `portfolio-engine-attachments` tự động chạy khi có commit chạm `data/portfolios/**` hoặc khi kích hoạt thủ công (`workflow_dispatch`).
- Pipeline setup Python 3.11, in ra toàn bộ CSV dưới `data/portfolios/` để bạn đối chiếu danh mục gốc, chạy `./broker.sh engine`, sau đó upload artifact `.artifacts/engine/attachments_latest.zip` với thời hạn lưu 3 ngày để bạn tải trực tiếp từ trang run.

## Lưu ý

- Thiếu `data/universe/vn100.csv` → engine dừng ngay.
- Thiếu cột bắt buộc trong CSV → raise lỗi rõ ràng.
- Nếu ATR14 hoặc ADV20 bằng 0 → gắn guard `ZERO_ATR`/`LOW_LIQ` nhưng vẫn xuất bảng.
- Khi giá điều chỉnh vượt biên → clamp và thêm guard `CLAMPED`, `NEAR_LIMIT` khi sát trần/sàn.
- Toàn bộ đường dẫn được ghi tương đối theo repo trong manifest/bundle để tiện attach vào ChatGPT.
