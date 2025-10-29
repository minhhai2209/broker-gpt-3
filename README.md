# Broker GPT Data Engine

> Công cụ này không đưa ra lời khuyên đầu tư. Nó chỉ thu thập dữ liệu giá, tính toán chỉ số kỹ thuật và ghi lại kết quả để bạn ra quyết định thủ công.

## Tổng quan

Data engine được thiết kế lại để làm đúng một việc: chuẩn bị dữ liệu sạch cho ChatGPT (hoặc bất kỳ công cụ phân tích nào khác) sử dụng. Mỗi lần chạy engine sẽ:

1. Thu thập dữ liệu lịch sử và intraday cho toàn bộ vũ trụ mã.
2. Tính toán sẵn các chỉ báo kỹ thuật cơ bản và ghi vào một file CSV duy nhất (`out/technical_snapshot.csv`).
3. Tính toán các mức giá mua/bán theo từng preset và xuất thành từng file CSV riêng (`out/preset_<preset>.csv`).
4. Đọc danh mục hiện có của từng tài khoản, tổng hợp lãi/lỗ rồi đóng gói cùng snapshot/preset thành bundle phẳng `out/bundle_<profile>.zip`.
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
  returns_periods: [20, 60]
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
presets:
  balanced:
    buy_tiers: [-0.03, -0.02, -0.01]
    sell_tiers: [0.02, 0.04, 0.06]
portfolio:
  directory: data/portfolios     # Mỗi tài khoản là 1 thư mục: <profile>/portfolio.csv
  order_history_directory: data/order_history
output:
  base_dir: out
  market_snapshot: technical_snapshot.csv
  presets_dir: .
  portfolios_dir: .
```

Bạn có thể chỉnh preset (tỷ lệ ± so với giá hiện tại), đường dẫn output hoặc bổ sung chỉ báo tuỳ nhu cầu.

### Shortlist cho presets (tuỳ chọn nhưng bật sẵn)

Để chỉ loại các mã “xấu hẳn” khỏi danh sách cân nhắc, engine hỗ trợ bộ lọc bảo thủ cho presets. Cấu hình tại `filters.shortlist` trong `config/data_engine.yaml`:

- `enabled`: bật/tắt shortlist.
- Điều kiện “rất yếu” (mặc định yêu cầu hội tụ tất cả):
  - `RSI_14` ≤ `rsi14_max` (mặc định 25)
  - `PctToLo_252` ≤ `max_pct_to_lo_252` (mặc định 2%) — giá sát đáy 52w
  - `Return_20` ≤ `return20_max` (mặc định -15%) và `Return_60` ≤ `return60_max` (mặc định -25%)
  - Giá dưới cả `SMA_50` và `SMA_200`
  - (tuỳ chọn) `ADV_20` ≤ `min_adv_20` để loại mã quá kém thanh khoản
- `drop_logic_all`: nếu `true` (mặc định), chỉ loại khi tất cả điều kiện cùng thoả.
- `keep`/`exclude`: danh sách mã luôn giữ lại/luôn loại bỏ (mặc định để trống). Ví dụ: có thể thêm `FPT` vào `keep` nếu muốn minh hoạ, nhưng không bật sẵn để tránh lọc quá “aggressive”.

Mục tiêu là giảm nhiễu, không làm nghèo vũ trụ cơ hội; vì vậy mặc định bộ lọc rất bảo thủ.

## Cách chạy

### Engine + Scraper (mặc định)

```bash
./broker.sh             # chạy TCBS (headful) rồi chạy engine
```

Chuỗi mặc định sẽ chạy:
- `tcbs --headful` để lấy danh mục (và lệnh khớp hôm nay, mặc định bật).
- `engine` để cập nhật snapshot kỹ thuật, preset và báo cáo danh mục.

### Engine (chạy riêng)

```bash
./broker.sh engine
```

Engine sẽ:

- Gọi API VNDIRECT để cập nhật giá lịch sử + intraday.
- Tính SMA/RSI/ATR/MACD theo cấu hình.
- Xuất các file CSV đã nêu ở trên.
- Xoá sạch thư mục `out/` trước khi chạy để tính toán lại toàn bộ (bao gồm preset/báo cáo danh mục). Sau khi chạy xong, engine đóng gói các file phẳng theo `prompts/PROMPT.txt` thành `out/bundle_<profile>.zip` (mỗi profile một file).

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
| `out/technical_snapshot.csv` | Bảng tổng hợp theo mã: giá hiện tại, thay đổi %, SMA/RSI/ATR/MACD, sector |
| (mở rộng) | EMA_*, ATRPct_*, Return_*, Z_*, Hi_252, Lo_252, PctFromHi_252, PctToLo_252, ADV_* |
| `out/preset_<preset>.csv` | Mỗi preset một file; chứa giá mua/bán theo từng bậc |
| `out/bundle_<profile>.zip` | Gói phẳng: `technical_snapshot.csv`, `preset_*.csv`, `portfolio.csv`, `positions.csv`, `sector.csv`, `fills.csv` |
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

Gợi ý kiểm tra nhanh (giá báo theo nghìn đồng):
- `p_vnd = round(LimitPrice * 1000)`, chọn `tick` theo bảng trên tại mức `p_vnd`.
- Hợp lệ khi `(p_vnd % tick == 0)` và `floor_to_tick(ref*1.07) ≥ p_vnd ≥ ceil_to_tick(ref*0.93)`; `ref` là giá tham chiếu VND.
- `Quantity` là bội số 100 và ≤ 500.000.
