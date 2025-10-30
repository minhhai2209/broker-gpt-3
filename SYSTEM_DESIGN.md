# Kiến trúc Data Engine (2025)

## Mục tiêu

Phiên bản này bỏ hoàn toàn order engine. Toàn bộ hệ thống chỉ còn các thành phần sau:

1. **Engine thu thập dữ liệu** (`scripts/engine/data_engine.py`): tải dữ liệu giá, tính chỉ số kỹ thuật, dựng bands/sizing/signals/limits và cập nhật báo cáo danh mục/sector. Khi khởi chạy, engine sẽ xoá sạch `out/`; kết thúc sẽ ghi 7 file CSV chuẩn hoá rồi đóng gói phẳng theo `prompts/PROMPT.txt` tại `out/bundle_<profile>.zip` (mỗi profile một file). Lưu ý: `levels.csv` đã bị loại khỏi output; `sizing.csv` bỏ `TargetQty/DeltaQty/SliceCount/SliceQty`; `signals.csv` chỉ giữ `Ticker,BandDistance`.
2. **Kho dữ liệu danh mục** (`data/portfolios/`, `data/order_history/`): lưu trữ danh mục hiện tại và lịch sử khớp lệnh của từng tài khoản.
3. **TCBS Scraper** (`scripts/scrapers/tcbs.py`): đăng nhập TCBS bằng Playwright, ghi `data/portfolios/<profile>/portfolio.csv` và mặc định thu thập các lệnh đã khớp trong hôm nay vào `data/order_history/<profile>/fills.csv` (kèm bản đầy đủ `fills_all.csv`). Có thể tắt bằng `--no-fills`.
4. (Tạm thời vô hiệu) GitHub Action: trước đây workflow tại `.github/workflows/data-engine.yml` chạy engine định kỳ và commit kết quả. Hiện đã gỡ; chạy local thay thế.

Mọi quyết định giao dịch sẽ do người vận hành xử lý dựa trên dữ liệu CSV đầu ra.

## Dòng dữ liệu

```
┌───────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│ data_engine.py│──►──│ out/*.csv & bundle   │────►│ ChatGPT / analyst UI │
└─────▲─────────┘     └────────────▲────────┘     └──────────▲───────────┘
      │                              │                           │
      │                              │                           │
      │            ┌─────────────────┴─────────────┐             │
      │            │ data/portfolios/*/portfolio.csv│◄────┐      │
      │            │ data/order_history/<profile>/fills.csv│      │      │
      │            └───────────────────────────────┘      │      │
      │                     ▲                            │      │
      │                     │                            │      │
      └────── tcbs.py ◄────┴────────── Fetch via browser ◄──────┘      │
```

- Engine đọc universe từ `config/data_engine.yaml` (tối thiểu cột `Ticker` và `Sector`).
- Engine thêm mọi mã đang có trong danh mục vào universe để chắc chắn có dữ liệu giá.
- Dữ liệu lịch sử và intraday lấy từ API VNDIRECT (module `collect_intraday` và `fetch_ticker_data`). Cache được lưu ở `out/data/`.
- Tất cả output CSV nằm dưới `out/`. Khi workflow bị gỡ, bạn cần tự commit/push khi có thay đổi.

## Thành phần chính

### EngineConfig

`EngineConfig.from_yaml(path)` đọc file YAML và chuẩn hoá:

- `universe.csv` – nguồn danh sách mã + sector.
- `technical_indicators` – cấu hình SMA/RSI/ATR/MACD.
- `portfolio.directory` – thư mục chứa danh mục từng tài khoản.
- `order_history_directory` – thư mục append lịch sử khớp lệnh.
- `output` – vị trí ghi các file CSV chuẩn hoá (default `out/`).
- `execution` – tham số sizing (aggressiveness, max_order_pct_adv, slice_adv_ratio, min lot, max qty/order).
- `data.history_cache` – nơi cache dữ liệu lịch sử.

Mọi đường dẫn được chuẩn hoá thành `Path.resolve()`. Thiếu trường bắt buộc sẽ raise `ConfigurationError` (fail-fast).

### VndirectMarketDataService

- `load_history(tickers)` gọi `ensure_and_load_history_df` để đảm bảo cache đầy đủ rồi trả về DataFrame hợp nhất (cột `Date,Ticker,Open,High,Low,Close,Volume,t`).
- `load_intraday(tickers)` gọi `ensure_intraday_latest_df` để lấy giá phút gần nhất. Nếu API fail, engine vẫn fallback về giá đóng cửa gần nhất.
- Có thể cung cấp `data/reference_overrides` (CSV `Ticker,Ref`) trong cấu hình để ép giá tham chiếu khi tính `bands.csv` trong các phiên có điều chỉnh tham chiếu của sàn.

### TechnicalSnapshotBuilder

- Ghép dữ liệu lịch sử và intraday, tính snapshot kỹ thuật chuẩn hoá.
- Với mỗi ticker:
  - `Last` = giá intraday nếu có, fallback `Close` cuối cùng; `Ref` = `LastClose`.
  - `ChangePct` = (Last/Ref − 1) ở dạng thập phân.
  - `SMA_20/50/200`, `EMA_20`, `RSI_14`, `ATR_14`, `MACD`, `MACDSignal`, `MACD_Hist`.
  - `Return_5`/`Return_20` (dạng phần trăm trong snapshot, được chuẩn hoá thành số thập phân khi ghi `technical.csv`).
  - `ADV_20`, `Hi_252`, `Lo_252`, Z-score (`Z_20`).
  - `Sector` lấy từ universe.
- Output trung gian giữ nguyên dưới dạng DataFrame; writer `_build_technical_output` chuyển thành `out/technical.csv` (một dòng/mã, chỉ giữ các cột chính nêu trong spec).

### PortfolioReporter

- Đọc từng danh mục `data/portfolios/<profile>/portfolio.csv` (schema: `Ticker,Quantity,AvgPrice`).
- Hợp nhất với snapshot để xác định `Last`, `Sector`, tính `MarketValue_kVND`, `CostBasis_kVND`, `Unrealized_kVND`, `PNLPct`.
- Sinh hai tầng kết quả:
  - Theo profile: DataFrame vị thế + tổng hợp sector (sử dụng khi ghi bundle `bundle_<profile>.zip`).
  - Tổng hợp toàn bộ profile: `aggregate_positions` (gộp theo ticker) và `aggregate_sector` (gộp theo sector) để ghi `out/positions.csv` và `out/sector.csv`.
- Không chạm vào file danh mục gốc; chỉ đọc.

### TCBS Scraper

- Đọc `TCBS_USERNAME` và `TCBS_PASSWORD` từ `.env` hoặc biến môi trường.
- Dùng Chromium persistent profile tại `.playwright/tcbs-user-data` để giữ fingerprint/session giữa các lần chạy (bỏ qua bước xác nhận thiết bị sau lần đầu).
- Điều hướng: login -> `my-asset` -> tab `Cổ phiếu` -> `Tài sản` -> bảng danh mục.
- Parse bảng một cách resilient theo header (`Mã`, `SL Tổng`/`Được GD`, `Giá vốn`) và ghi `data/portfolios/<profile>/portfolio.csv`.

## Quy trình chạy GitHub Action

Workflow (đã gỡ tạm thời):

1. Checkout mã nguồn (fetch đầy đủ lịch sử để có thể push).
2. Cài đặt Python 3.11 và dependencies (`pip install -r requirements.txt`).
3. Chạy `python -m scripts.engine.data_engine --config config/data_engine.yaml`.
4. (Trước đây) Nếu chạy theo lịch hoặc kích hoạt thủ công, workflow sẽ commit và push những thay đổi trong `out/*.csv`, `out/bundle_*.zip`, `out/diagnostics`, `data/order_history`. Khi chạy trên Pull Request, bước commit được bỏ qua để workflow chỉ dùng cho việc xem log.

Không còn workflow chạy định kỳ trong repo hiện tại. Nếu cần bật lại, thêm file YAML workflow vào `.github/workflows/`.

## Danh mục & lịch sử khớp lệnh

- Mỗi tài khoản → một thư mục `data/portfolios/<profile>/portfolio.csv` với schema tối thiểu `Ticker,Quantity,AvgPrice`.
- Lịch sử khớp lệnh ghi vào `data/order_history/<profile>/fills.csv`. Engine không xoá, server chỉ append.
- Khi engine chạy, file danh mục không bị sửa; các báo cáo được đóng gói thành `out/bundle_<profile>.zip` và có thể được ghi đè mỗi lần chạy.

## Kiểm thử

- `tests/test_data_engine.py` tạo dữ liệu giả, chạy engine và xác minh tất cả output tồn tại.
- `tests/test_tcbs_parser.py` xác minh bộ phân tích bảng TCBS với dữ liệu giả lập (không gọi mạng/trình duyệt).

## Mở rộng

- Có thể bổ sung chỉ báo mới bằng cách thêm vào `scripts/indicators/` và cập nhật `TechnicalSnapshotBuilder`.
- Nếu cần nguồn dữ liệu khác, triển khai class mới implement `MarketDataService` rồi truyền vào `DataEngine` (ví dụ trong test).
- Để đồng bộ với hệ thống khác, bạn chỉ cần đọc các CSV trong `out/` (được commit sẵn) hoặc pull nhánh mới nhất từ repo.
