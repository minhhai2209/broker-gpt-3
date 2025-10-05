Broker GPT — Giới thiệu & Sử dụng

Lưu ý: Đây không phải lời khuyên đầu tư. Công cụ chỉ hỗ trợ ra quyết định có kỷ luật cho thị trường Việt Nam.

Phạm vi & giả định
- Engine mặc định tối ưu cho rổ VN100 (HOSE) và giả định danh mục hiện có ít nhất 1 mã HOSE. Nếu danh mục trống hoặc chứa mã ngoài HOSE, cần cập nhật dữ liệu/luật giao dịch trước khi dùng.

Mục tiêu README
- Tập trung vào giới thiệu repo và cách sử dụng nhanh. Toàn bộ kiến trúc, thuật toán, calibrations… được trình bày chi tiết tại SYSTEM_DESIGN.md.

Yêu cầu hệ thống
- Python 3.10+ (khuyến nghị 3.11)
- macOS/Linux/WSL (terminal)

Cài đặt
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Chuẩn bị danh mục (inputs)
- Thư mục: `in/portfolio/`
- Hỗ trợ 1 hoặc nhiều CSV cùng schema cơ bản:
  - `Ticker,Quantity,AvgCost` (đơn vị `AvgCost` là nghìn đồng/cp)
- Tự chuẩn hoá mã (upper/strip, loại kí tự phụ như `*`). Nhiều file sẽ được hợp nhất theo mã (cộng `Quantity`, bình quân gia quyền `AvgCost`).
- Tạo nhanh từ mẫu:
```bash
mkdir -p in/portfolio
cp samples/portfolio.csv in/portfolio/portfolio.csv
```

Chạy nhanh (tạo lệnh)
- Lệnh mặc định: `./broker.sh` (tương đương `./broker.sh orders`).
- Script sẽ dựng pipeline cần thiết và xuất kết quả vào `out/`.

Kết quả chính (out/)
- `out/orders/orders_final.csv` — file để nhập lệnh: `Ticker,Side,Quantity,LimitPrice` (BUY trước SELL; chỉ 4 cột tối giản).
- `out/orders/orders_watchlist.csv` — các lệnh BUY bị đẩy ra watchlist do tín hiệu/yếu tố vi mô.
- `out/orders/orders_quality.csv` — bảng tham chiếu giàu thông tin (MarketPrice, FillProb, FillRateExp, ExpR, Priority, TTL_Min, SlipBps/Pct, Signal, LimitLock, Notes).
- `out/portfolio_evaluation.txt|.csv` — đánh giá danh mục: phân bổ ngành, HHI/top‑N, thanh khoản, ATR%/Beta.
- Các file hỗ trợ khác có thể xuất hiện: `orders_print.txt`, `orders_reasoning.csv`, `orders_analysis.txt`, `trade_suggestions.txt`.

Lệnh tiện ích
- `./broker.sh orders` — chạy Order Engine (mặc định).
- `./broker.sh tests` — chạy test; bật coverage: `BROKER_COVERAGE=1 ./broker.sh tests`.
- `./broker.sh policy` — sinh `config/policy_overrides.json` (guardrails + commit/push) để dùng cùng `config/policy_default.json`.
- `./broker.sh server` — chạy API server cục bộ (Flask) phục vụ extension/ứng dụng (mặc định `PORT=8787`).

Backtest & tuning (MVP)
- `python -m backtest.runner --start YYYY-MM-DD --end YYYY-MM-DD --config configs/replay.yaml --out backtest_out/<run_id>` để replay một cửa sổ lịch sử và sinh CSV/NDJSON phục vụ audit.
- `python -m backtest.tune --train-start YYYY-MM-DD --train-end YYYY-MM-DD --val-start YYYY-MM-DD --val-end YYYY-MM-DD --grid configs/tune_grid.yaml --base-config configs/replay.yaml` để chạy grid search đơn giản; tham số walk-forward cũng có sẵn qua các flag `--wf-*` theo spec.
- GitHub Actions mặc định chỉ chạy `./broker.sh tests` (xem `.github/workflows/tests.yml`). Khi cần chuỗi đầy đủ (tests → tuning AI overrides → nightly calibrations → tạo lệnh từ danh mục cuối cùng), kích hoạt workflow thủ công **Full Pipeline Orchestration** và chọn `orders-selection=latest` để dùng danh mục mới nhất.

Calibrate TTL (Time-To-Live)
- TTL mặc định trong policy là `base/soft/hard = 12/9/7` phút. Khi có dữ liệu VNINDEX mới, sử dụng biến động Garman–Klass để co giãn TTL nhằm phản ánh trạng thái thị trường hiện tại.
- Script hỗ trợ: `python scripts/engine/calibrate_ttl_minutes.py`. Script đọc `out/prices_history.csv` + `out/orders/policy_overrides.json`, tính bucket biến động (low/medium/high) và cập nhật `orders_ui.ttl_minutes` cùng metadata (`ttl_bucket_minutes`, `ttl_bucket_thresholds`, `ttl_bucket_state`).
- Bucket → TTL: low → 14/11/8 phút, medium → 11/9/7 phút, high → 8/6/5 phút. TTL viết lại vào `policy_overrides.json`; lần chạy kế tiếp của Order Engine sẽ sử dụng TTL mới.

API server (tùy chọn)
- Chạy: `./broker.sh server` → http://localhost:8787
- Endpoint chính:
  - `GET /health` — kiểm tra sống.
  - `POST /portfolio/reset` — xoá các CSV trong `in/portfolio/`.
  - `POST /portfolio/upload` — nạp 1 CSV (JSON body: `{name, content}`).
  - `POST /done` — bỏ qua policy và chỉ chạy Order Engine; trả về đường dẫn các file lệnh trong `out/orders/`.
- Ghi chú: `/done` hiện luôn bỏ qua bước policy; policy có thể được cập nhật riêng qua `./broker.sh policy` hoặc bởi PolicyScheduler nếu bật.

Policy & cấu hình
- Baseline: `config/policy_default.json` (nguồn sự thật, có chú thích đầy đủ).
- Overrides cho phiên/ngày: `config/policy_overrides.json` — chỉ cho phép các khoá whitelisted (xem chi tiết và guardrails trong SYSTEM_DESIGN.md).
- Nếu không có overrides, engine dùng nguyên baseline.

Execution diagnostics & chi phí giao dịch
- Engine ước lượng xác suất khớp và tỷ lệ khớp kỳ vọng dựa trên thanh khoản (ADTV) để tránh giả định “touch là khớp”. Các giá trị này xuất hiện ở `orders_quality.csv` (`FillProb`, `FillRateExp`).
- Slippage và chi phí giao dịch được nội suy qua `pricing.tc_roundtrip_frac` và mô hình trượt giá tuyến tính; kết quả thể hiện dưới `SlipBps`, `SlipPct` và được phản ánh trong `ExpR` (Expected Return sau phí).
- `orders_quality.csv` bổ sung cờ `LimitLock` nhằm cảnh báo các phiên trần/sàn (lock-limit day) mà BUY/SELL khó khớp.

Tài liệu chi tiết
- Kiến trúc, pipeline, nhận diện market regime, thuật toán quyết định lệnh, calibrations, guardrails I/O: xem `SYSTEM_DESIGN.md`.

FAQ nhanh / khắc phục sự cố
- Danh mục rỗng hoặc sai đơn vị giá vốn → kiểm tra `in/portfolio/*.csv` (AvgCost tính theo nghìn đồng/cp), xem `out/portfolio_clean.csv` để xác nhận ingest.
- Thiếu dữ liệu lịch sử → chạy lại `./broker.sh orders` (engine tự dựng cache trong `out/data/`).
- Muốn audit thêm lý do/điểm số → xem `out/orders/orders_reasoning.csv` và `out/orders/orders_quality.csv`.

Góp ý & đóng góp
- Mở issue/PR nếu phát hiện lỗi tài liệu/UX; nội dung chuyên sâu xin bổ sung vào `SYSTEM_DESIGN.md` để tránh lặp lại trong README.

