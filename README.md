# Broker GPT Data Engine

> Công cụ này không đưa ra lời khuyên đầu tư. Nó chỉ thu thập dữ liệu giá, tính toán chỉ số kỹ thuật và ghi lại kết quả để bạn ra quyết định thủ công.

## Tổng quan

Data engine được thiết kế lại để làm đúng một việc: chuẩn bị dữ liệu sạch cho ChatGPT (hoặc bất kỳ công cụ phân tích nào khác) sử dụng. Mỗi lần chạy engine sẽ:

1. Thu thập dữ liệu lịch sử và intraday cho toàn bộ vũ trụ mã.
2. Chuẩn hoá snapshot kỹ thuật (`out/technical.csv`) với SMA/EMA/RSI/ATR/MACD, lợi suất và biên độ 52w.
3. Tính biên trần/sàn, sizing và tín hiệu phụ trợ + thông số vận hành (`out/bands.csv`, `out/sizing.csv`, `out/signals.csv`, `out/limits.csv`).
4. Làm giàu danh mục/sector theo giá hiện tại (`out/positions.csv`, `out/sector.csv`) và đóng gói 8 file phẳng vào `out/bundle_<profile>.zip`.
5. Giữ nguyên lịch sử khớp lệnh dạng CSV trong `data/order_history/` (không xoá).

Không còn bước tạo lệnh tự động, không còn phụ thuộc Vietstock, không còn overlay policy. Bạn chủ động đọc các file CSV và đưa ra quyết định.

## Chuẩn bị môi trường

- Python 3.10 trở lên.
- macOS, Linux hoặc WSL đều chạy được.
- Khuyến nghị tạo virtualenv trước khi chạy.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Cấu hình engine

File chính: `config/data_engine.yaml`

```yaml
universe:
  csv: data/industry_map.csv    # Danh sách mã + sector
technical_indicators:
  moving_averages: [20, 50, 200]
  ema_periods: [20, 50]
  rsi_periods: [14]
  atr_periods: [14]
  returns_periods: [5, 20, 60]
  bollinger:
    windows: [20]
    k: 2
    include_bands: false
  range_lookback_days: 252
  adv_periods: [20]
  macd:
    fast: 12
    slow: 26
    signal: 9
portfolio:
  directory: data/portfolios     # Mỗi tài khoản là 1 thư mục: <profile>/portfolio.csv
  order_history_directory: data/order_history
output:
  base_dir: out
  presets_dir: .
  portfolios_dir: .
execution:
  aggressiveness: med
  max_order_pct_adv: 0.1
  slice_adv_ratio: 0.25
  min_lot: 100
  max_qty_per_order: 500000
data:
  history_cache: out/data
  history_min_days: 1
  intraday_window_minutes: 60
  # Optional: override HOSE reference prices when có điều chỉnh tham chiếu
  # reference_overrides: data/reference_overrides.csv   # schema: Ticker,Ref
```

Bạn có thể tinh chỉnh tham số chỉ báo, đường dẫn output hoặc giới hạn sizing (`execution`) tuỳ nhu cầu.

## Cách chạy

### Engine + Scraper (mặc định)

```bash
./broker.sh             # chạy TCBS (headful) rồi chạy engine
```

Chuỗi mặc định sẽ chạy:
- `tcbs --headful` để lấy danh mục (và lệnh khớp hôm nay, mặc định bật).
- `engine` để cập nhật snapshot kỹ thuật và bộ file output (bands/sizing/signals/limits/positions/sector).

### Engine (chạy riêng)

```bash
./broker.sh engine
```

Engine sẽ:

- Gọi API VNDIRECT để cập nhật giá lịch sử + intraday.
- Tính SMA/RSI/ATR/MACD theo cấu hình.
- Xuất 7 file CSV chuẩn hoá: `technical.csv`, `bands.csv`, `sizing.csv`, `signals.csv`, `limits.csv`, `positions.csv`, `sector.csv`.
- Xoá sạch thư mục `out/` trước khi chạy để tính toán lại toàn bộ. Sau khi chạy xong, engine đóng gói 8 file này thành `out/bundle_<profile>.zip` (mỗi profile một file, không thêm portfolio/fills).

### Lấy danh mục + lệnh khớp hôm nay (TCBS, Playwright)

Thay cho server HTTP, repo cung cấp scraper Playwright để đăng nhập TCBS và trích xuất danh mục.

Chuẩn bị:
- Tạo file `.env` ở repo root:

```
TCBS_USERNAME=you-username
TCBS_PASSWORD=your-password-here
TCBS_PROFILE=your-profile
```

Chạy lần đầu (headful để xác nhận thiết bị nếu TCBS yêu cầu):

```bash
./broker.sh tcbs --headful
```

Chạy các lần sau (headless, fingerprint được lưu trong `.playwright/tcbs-user-data`):

```bash
./broker.sh tcbs
```

Mặc định script sẽ:
- Ghi đè `data/portfolios/<profile>/portfolio.csv` với cột `Ticker,Quantity,AvgPrice`.
- Lấy các lệnh đã khớp trong hôm nay và ghi `data/order_history/<profile>/fills.csv` (chỉ hôm nay) và `data/order_history/<profile>/fills_all.csv` (đã chuẩn hoá, đầy đủ lịch sử bảng Tra cứu).

Tắt lấy “fills” nếu cần:

```bash
./broker.sh tcbs --no-fills
```

### Kiểm thử

```bash
./broker.sh tests
```

Test bao gồm:
- Bảo đảm engine sinh đầy đủ output khi dùng nguồn dữ liệu giả lập.
- Xác thực bộ phân tích bảng TCBS tạo đúng schema `Ticker,Quantity,AvgPrice` từ dữ liệu giả lập.

## Output chính

| File | Ý nghĩa |
| ---- | ------- |
| `out/technical.csv` | Snapshot kỹ thuật chuẩn hoá: Last/Ref, SMA20/50/200, EMA20, RSI14, ATR14, MACD, Ret5d/Ret20d, ADV20, 52w range |
| `out/bands.csv` | Tick hợp lệ và giá trần/sàn HOSE theo bước giá chuẩn |
| `out/sizing.csv` | Điểm thanh khoản/biến động và giới hạn lệnh theo mã (không còn TargetQty/DeltaQty/SliceCount/SliceQty) |
| `out/signals.csv` | Tín hiệu cơ bản: chỉ còn BandDistance (PresetFit/SectorBias/RiskGuards đã loại bỏ) |
| `out/limits.csv` | Tham số vận hành engine: aggressiveness, max_order_pct_adv, slice_adv_ratio, min_lot, max_qty_per_order |
| `out/positions.csv` | Danh mục hiện tại đã enrich: MarketValue_kVND, CostBasis_kVND, Unrealized_kVND, PNLPct |
| `out/sector.csv` | Tổng hợp giá trị/PNL theo ngành và trọng số trong danh mục |
| `out/bundle_<profile>.zip` | Gói phẳng đúng 7 file trên cho từng profile (không đính kèm portfolio hoặc fills) |
| `data/order_history/<profile>/fills.csv` | Lệnh khớp hôm nay (do scraper TCBS ghi) |
| `data/order_history/<profile>/fills_all.csv` | Bảng lệnh khớp đã chuẩn hoá đầy đủ |

## GitHub Actions

Hiện tại GitHub Action đã được tạm thời gỡ khỏi repo. Hãy chạy engine thủ công trên máy local bằng `./broker.sh engine`. Khi cần bật lại, thêm lại workflow vào `.github/workflows/` (xem lịch sử commit trước đó để tham khảo cấu hình).

## Hỏi nhanh

**Có cần sửa danh mục thủ công?** — Có. Mỗi tài khoản là một thư mục trong `data/portfolios/` chứa `portfolio.csv`. Engine chỉ đọc và ghi báo cáo, không can thiệp vào file gốc.

**Lịch sử khớp lệnh lưu ở đâu?** — `data/order_history/<profile>/fills.csv` (hôm nay) và `data/order_history/<profile>/fills_all.csv` (đầy đủ) do scraper TCBS ghi. Có thể tắt bằng `--no-fills`.

**Muốn thêm chỉ báo mới?** — Bổ sung vào `scripts/indicators/` hoặc tính trực tiếp trong `scripts/engine/data_engine.py`, sau đó khai báo trong `config/data_engine.yaml` nếu cần tham số.

**Có còn chính sách/overlay?** — Không. Engine không sinh lệnh nên mọi cấu hình policy trước đây đã bị loại bỏ.

## Prompt gợi ý cho ChatGPT

- Prompt chuẩn: `prompts/PROMPT.txt` — plain text, liệt kê đúng các file phẳng trong bundle.
- Xem nhanh/copy prompt: `./broker.sh prompts` (in đường dẫn), `./broker.sh prompts --outdir tmp` (copy thành `tmp/prompt.txt`) hoặc `./broker.sh prompts --dest my_prompt.txt`.

Ghi chú: Mô tả preset (balanced, momentum) đã được hard-code ngay trong prompt.

### Quy tắc HOSE để tính giá/khối lượng hợp lệ

Áp dụng đúng quy chế HOSE (QĐ 352/QĐ-SGDHCM, hiệu lực 05/07/2021) để tránh bước giá không hợp lệ:

- Đơn vị yết giá (tick) cổ phiếu/CCQ đóng:
  - < 10.000 VND: bước 10 VND
  - 10.000 – 49.950 VND: bước 50 VND
  - ≥ 50.000 VND: bước 100 VND
- ETF và chứng quyền: bước 10 VND cho mọi mức giá.
- Lô chẵn: bội số 100 cổ phiếu; khối lượng tối đa mỗi lệnh: 500.000 cổ.
- Biên độ giá trong ngày HOSE: ±7% so với giá tham chiếu.
- Làm tròn giá trần/sàn theo quy chế: trần làm tròn xuống, sàn làm tròn lên theo đúng đơn vị yết giá.

Lưu ý về giá tham chiếu: `bands.csv` tính từ cột `Ref` trong `technical.csv` (mặc định là giá đóng cửa gần nhất). Khi sàn điều chỉnh tham chiếu do cổ tức/CP thưởng/gộp tách…, hãy cung cấp `data/reference_overrides.csv` và khai báo `data.reference_overrides` như trên để engine dùng đúng giá tham chiếu của sàn.

Gợi ý kiểm tra nhanh (giá báo theo nghìn đồng):
- `p_vnd = round(LimitPrice * 1000)`, chọn `tick` theo bảng trên tại mức `p_vnd`.
- Hợp lệ khi `(p_vnd % tick == 0)` và `floor_to_tick(ref*1.07) ≥ p_vnd ≥ ceil_to_tick(ref*0.93)`; `ref` là giá tham chiếu VND.
- `Quantity` là bội số 100 và ≤ 500.000.
