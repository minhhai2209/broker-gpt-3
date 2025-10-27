# Kiến trúc Data Engine (2025)

## Mục tiêu

Phiên bản này bỏ hoàn toàn order engine. Toàn bộ hệ thống chỉ còn các thành phần sau:

1. **Engine thu thập dữ liệu** (`scripts/engine/data_engine.py`): tải dữ liệu giá, tính chỉ số kỹ thuật, sinh preset và cập nhật báo cáo danh mục.
2. **Kho dữ liệu danh mục** (`data/portfolios/`, `data/order_history/`): lưu trữ danh mục hiện tại và lịch sử khớp lệnh của từng tài khoản.
3. **API upload** (`scripts/api/server.py`): nhận dữ liệu danh mục từ client và ghi vào kho.
4. **GitHub Action** (`.github/workflows/data-engine.yml`): chạy engine định kỳ và commit kết quả mới lên nhánh hiện hành.

Mọi quyết định giao dịch sẽ do người vận hành xử lý dựa trên dữ liệu CSV đầu ra.

## Dòng dữ liệu

```
┌───────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│ data_engine.py│──►──│ out/market/*.csv     │────►│ ChatGPT / analyst UI │
└─────▲─────────┘     └────────────▲────────┘     └──────────▲───────────┘
      │                              │                           │
      │                              │                           │
      │            ┌─────────────────┴─────────────┐             │
      │            │ data/portfolios/*.csv         │◄─────┐      │
      │            │ data/order_history/*_fills.csv│      │      │
      │            └───────────────────────────────┘      │      │
      │                     ▲                            │      │
      │                     │                            │      │
      └────── server.py ◄───┴────────── Upload JSON ◄────┘      │
```

- Engine đọc universe từ `config/data_engine.yaml` (tối thiểu cột `Ticker` và `Sector`).
- Engine thêm mọi mã đang có trong danh mục vào universe để chắc chắn có dữ liệu giá.
- Dữ liệu lịch sử và intraday lấy từ API VNDIRECT (module `collect_intraday` và `fetch_ticker_data`). Cache được lưu ở `out/data/`.
- Tất cả output CSV nằm dưới `out/` và được workflow commit/push khi có thay đổi.

## Thành phần chính

### EngineConfig

`EngineConfig.from_yaml(path)` đọc file YAML và chuẩn hoá:

- `universe.csv` – nguồn danh sách mã + sector.
- `technical_indicators` – cấu hình SMA/RSI/ATR/MACD.
- `presets` – mỗi preset gồm `buy_tiers` và `sell_tiers` (tỷ lệ so với giá hiện tại).
- `portfolio.directory` – thư mục chứa danh mục từng tài khoản.
- `order_history_directory` – thư mục append lịch sử khớp lệnh.
- `output` – vị trí ghi market snapshot, preset và báo cáo danh mục.
- `data.history_cache` – nơi cache dữ liệu lịch sử.

Mọi đường dẫn được chuẩn hoá thành `Path.resolve()`. Thiếu trường bắt buộc sẽ raise `ConfigurationError` (fail-fast).

### VndirectMarketDataService

- `load_history(tickers)` gọi `ensure_and_load_history_df` để đảm bảo cache đầy đủ rồi trả về DataFrame hợp nhất (cột `Date,Ticker,Open,High,Low,Close,Volume,t`).
- `load_intraday(tickers)` gọi `ensure_intraday_latest_df` để lấy giá phút gần nhất. Nếu API fail, engine vẫn fallback về giá đóng cửa gần nhất.

### TechnicalSnapshotBuilder

- Ghép dữ liệu lịch sử và intraday, tính các chỉ số kỹ thuật.
- Với mỗi ticker:
  - `LastPrice` = intraday nếu có, ngược lại lấy `Close` cuối cùng.
  - `ChangePct` = phần trăm so với phiên trước.
  - `SMA_<n>`/`RSI_<n>`/`ATR_<n>`/`MACD_Hist` theo config.
  - `Sector` lấy từ universe.
- Output: `out/market/technical_snapshot.csv` (một dòng/mã, đầy đủ các cột kỹ thuật và thời gian cập nhật).

### PresetWriter

- Đọc snapshot, tạo file cho từng preset dưới `out/presets/`.
- Mỗi file gồm `Ticker`, `Sector`, `LastPrice`, `LastClose`, `PriceSource`, các cột `Buy_i`, `Sell_i` (round 4 chữ số).
- Nếu preset có mô tả (`description`), cột `PresetDescription` được thêm vào.

### PortfolioReporter

- Đọc từng file danh mục `data/portfolios/<profile>.csv` (schema: `Ticker,Quantity,AvgPrice`).
- Hợp nhất với snapshot để xác định `LastPrice` và sector.
- Tính toán:
  - `MarketValue`, `CostBasis`, `UnrealizedPnL`, `UnrealizedPct`.
- Ghi `out/portfolios/<profile>_positions.csv`.
- Tổng hợp theo ngành -> `out/portfolios/<profile>_sector.csv`.
- Không chạm vào file danh mục gốc; chỉ đọc.

### PortfolioStorage (API server)

- Endpoint duy nhất `POST /upload` nhận JSON `{profile, portfolio: [...], fills: [...]}`.
- `profile` được làm sạch chỉ giữ lại ký tự chữ, số, `_` hoặc `-`.
- `portfolio` phải có cột `Ticker,Quantity,AvgPrice`; server ghi đè file `data/portfolios/<profile>.csv`.
- `fills` (nếu có) được append vào `data/order_history/<profile>_fills.csv` với header tự động.
- Trả về `{status:"ok", profile, portfolio_path, fills_path?}`.

## Quy trình chạy GitHub Action

Workflow `.github/workflows/data-engine.yml`:

1. Checkout mã nguồn (fetch đầy đủ lịch sử để có thể push).
2. Cài đặt Python 3.11 và dependencies (`pip install -r requirements.txt`).
3. Chạy `python -m scripts.engine.data_engine --config config/data_engine.yaml`.
4. Nếu workflow chạy theo lịch hoặc được kích hoạt thủ công trên nhánh chính, commit và push những thay đổi trong `out/market`, `out/presets`, `out/portfolios`, `out/diagnostics`, `data/order_history`. Khi chạy trên Pull Request, bước commit được bỏ qua để workflow chỉ dùng cho việc xem log.

Không còn workflow tuning/policy. Nếu cần cập nhật config, commit trực tiếp file YAML.

## Danh mục & lịch sử khớp lệnh

- Mỗi tài khoản → một file CSV `data/portfolios/<profile>.csv` với schema tối thiểu `Ticker,Quantity,AvgPrice`.
- Lịch sử khớp lệnh ghi vào `data/order_history/<profile>_fills.csv`. Engine không xoá, server chỉ append.
- Khi engine chạy, file danh mục không bị sửa; các báo cáo nằm ở `out/portfolios/` và có thể được ghi đè mỗi lần chạy.

## Kiểm thử

- `tests/test_data_engine.py` tạo dữ liệu giả, chạy engine và xác minh tất cả output tồn tại.
- `tests/test_server.py` khởi tạo server với config tạm, gửi POST `/upload` và kiểm tra file sinh ra.

## Mở rộng

- Có thể bổ sung chỉ báo mới bằng cách thêm vào `scripts/indicators/` và cập nhật `TechnicalSnapshotBuilder`.
- Nếu cần nguồn dữ liệu khác, triển khai class mới implement `MarketDataService` rồi truyền vào `DataEngine` (ví dụ trong test).
- Để đồng bộ với hệ thống khác, bạn chỉ cần đọc các CSV trong `out/` (được commit sẵn) hoặc pull nhánh mới nhất từ repo.

