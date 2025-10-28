# Broker GPT Data Engine

> Công cụ này không đưa ra lời khuyên đầu tư. Nó chỉ thu thập dữ liệu giá, tính toán chỉ số kỹ thuật và ghi lại kết quả để bạn ra quyết định thủ công.

## Tổng quan

Data engine được thiết kế lại để làm đúng một việc: chuẩn bị dữ liệu sạch cho ChatGPT (hoặc bất kỳ công cụ phân tích nào khác) sử dụng. Mỗi lần chạy engine sẽ:

1. Thu thập dữ liệu lịch sử và intraday cho toàn bộ vũ trụ mã.
2. Tính toán sẵn các chỉ báo kỹ thuật cơ bản và ghi vào một file CSV duy nhất (`out/market/technical_snapshot.csv`).
3. Tính toán các mức giá mua/bán theo từng preset và xuất thành từng file CSV riêng (`out/presets/<preset>.csv`).
4. Đọc danh mục hiện có của từng tài khoản, cập nhật lãi/lỗ theo mã và theo ngành vào `out/portfolios/*.csv`.
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
  directory: data/portfolios     # Mỗi tài khoản 1 CSV: Ticker,Quantity,AvgPrice
  order_history_directory: data/order_history
output:
  base_dir: out
  market_snapshot: market/technical_snapshot.csv
  presets_dir: presets
  portfolios_dir: portfolios
```

Bạn có thể chỉnh preset (tỷ lệ ± so với giá hiện tại), đường dẫn output hoặc bổ sung chỉ báo tuỳ nhu cầu.

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
- Ghi đè `data/portfolios/<profile>.csv` với cột `Ticker,Quantity,AvgPrice`.
- Lấy các lệnh đã khớp trong hôm nay và ghi `data/order_history/<profile>_fills.csv` (chỉ hôm nay) và `data/order_history/<profile>_fills_all.csv` (đã chuẩn hoá, đầy đủ lịch sử bảng Tra cứu).

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
| `out/market/technical_snapshot.csv` | Bảng tổng hợp theo mã: giá hiện tại, thay đổi %, SMA/RSI/ATR/MACD, sector |
| (mở rộng) | EMA_*, ATRPct_*, Return_*, Z_*, Hi_252, Lo_252, PctFromHi_252, PctToLo_252, ADV_* |
| `out/presets/<preset>.csv` | Mỗi preset một file; chứa giá mua/bán theo từng bậc |
| `out/portfolios/<profile>_positions.csv` | Phân tích lãi/lỗ theo mã cho danh mục `profile` |
| `out/portfolios/<profile>_sector.csv` | Tổng hợp lãi/lỗ theo ngành |
| `data/order_history/<profile>_fills.csv` | Lệnh khớp hôm nay (do scraper TCBS ghi) |
| `data/order_history/<profile>_fills_all.csv` | Bảng lệnh khớp đã chuẩn hoá đầy đủ |

## GitHub Actions

Repo chỉ còn một workflow: `.github/workflows/data-engine.yml`. Workflow này chạy 10 phút/lần, luôn có thể kích hoạt thủ công (`workflow_dispatch`), và cũng tự động chạy trên mọi Pull Request để dễ dàng xem log trước khi merge. Sau khi engine hoàn tất, workflow sẽ commit và push trực tiếp các thư mục `out/market`, `out/presets`, `out/portfolios`, `out/diagnostics` cùng với `data/order_history` vào nhánh hiện hành (nếu có thay đổi, chỉ áp dụng cho chạy theo lịch/thủ công). Lưu ý: scraper TCBS chạy thủ công trên máy cá nhân (cần trình duyệt), không chạy trong CI.

## Hỏi nhanh

**Có cần sửa danh mục thủ công?** — Có. Mỗi tài khoản là một CSV trong `data/portfolios/`. Engine chỉ đọc và ghi báo cáo, không can thiệp vào file gốc.

**Lịch sử khớp lệnh lưu ở đâu?** — `data/order_history/<profile>_fills.csv` (hôm nay) và `data/order_history/<profile>_fills_all.csv` (đầy đủ) do scraper TCBS ghi. Có thể tắt bằng `--no-fills`.

**Muốn thêm chỉ báo mới?** — Bổ sung vào `scripts/indicators/` hoặc tính trực tiếp trong `scripts/engine/data_engine.py`, sau đó khai báo trong `config/data_engine.yaml` nếu cần tham số.

**Có còn chính sách/overlay?** — Không. Engine không sinh lệnh nên mọi cấu hình policy trước đây đã bị loại bỏ.

## Prompt gợi ý cho ChatGPT

- Template: `prompts/SAMPLE_PROMPT.txt` — plain text, chỉ có một placeholder `{{PROFILE}}`.
- Sinh prompt theo từng profile: `./broker.sh prompts` (quét `data/portfolios/*.csv` và tạo `prompts/prompt_<profile>.txt`).
- Sinh cho profile cụ thể: `./broker.sh prompts --profiles alpha,beta`.

Ghi chú: Mô tả preset (balanced, momentum) đã được hard-code ngay trong template.

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
