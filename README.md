Broker GPT — Giới thiệu & Sử dụng

Lưu ý: Đây không phải lời khuyên đầu tư. Công cụ chỉ hỗ trợ ra quyết định có kỷ luật cho thị trường Việt Nam.

Phạm vi & giả định
- Engine mặc định tối ưu cho rổ VN100 (HOSE) và giả định danh mục hiện có ít nhất 1 mã HOSE. Nếu danh mục trống hoặc chứa mã ngoài HOSE, cần cập nhật dữ liệu/luật giao dịch trước khi dùng.

Mục tiêu README
- Tập trung vào giới thiệu repo và cách sử dụng nhanh. Toàn bộ kiến trúc, thuật toán, calibrations… được trình bày chi tiết tại SYSTEM_DESIGN.md.

Yêu cầu hệ thống
- Python 3.10+ (khuyến nghị 3.11)
- macOS/Linux/WSL (terminal)
- (Chỉ cho lệnh `tune`/`policy`) Node.js 18+ với npm để cài Codex CLI

Cài đặt
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

- Nếu dự định dùng `tune`/`policy`: chạy thêm `npm install` (một lần) để bootstrap Codex CLI và cấu hình `~/.codex/config.toml` qua postinstall.

Ghi chú Codex CLI (cho `tune`/`policy`)
- Repo khai báo `@openai/codex` và có script postinstall (`scripts/postinstall-codex-global.js`). Khi chạy `npm install`:
  - Kiểm tra/cài Codex CLI toàn cục (`npm install -g @openai/codex@latest`), fallback `NPM_CONFIG_PREFIX=$HOME/.npm-global` nếu cần.
  - Sao chép `.codex/config.toml` từ repo → `~/.codex/config.toml` và đặt quyền `0600`.
  - Thiếu `.codex/config.toml` trong repo → in `::error::` và thoát `exit 2` (fail‑fast, phù hợp CI policy).
  - Nếu biến `CODEX_AUTH_JSON` có mặt, ghi `~/.codex/auth.json` (0600). Nếu job bắt buộc auth mà thiếu biến này, CI step sẽ fail.
- Tuner yêu cầu `codex` có trên PATH; nếu không có sau postinstall, script sẽ fail‑fast và in hướng dẫn bổ sung `PATH`.
- Nếu chạy cục bộ mà thiếu Node/npm, các lệnh cần Codex sẽ fail‑fast; cài Node.js (>=18) rồi chạy `npm install` một lần để bootstrap.

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
- `./broker.sh tune` — chạy calibrators + AI (Codex). Policy runtime ghi ở `out/orders/policy_overrides.json`. Khi publish, GitHub Action trên nhánh `main` sẽ đồng bộ sang `config/policy_overrides.json`.
- `./broker.sh server` — chạy API server cục bộ (Flask) phục vụ extension/ứng dụng (mặc định `PORT=8787`). Lưu ý: server luôn tắt auto‑policy (PolicyScheduler). Việc refresh policy được thực hiện bởi CI hoặc lệnh `./broker.sh policy` khi cần.

  

Diagnostics & calibrations
- Chi tiết về mô hình chi phí giao dịch, slippage, xác suất khớp (FillProb/FillRateExp), LimitLock, và cơ chế hiệu chỉnh TTL theo biến động thị trường được tài liệu hóa tại `SYSTEM_DESIGN.md` (mục Calibrations & Execution Diagnostics).

API server (tùy chọn)
- Chạy: `./broker.sh server` → http://localhost:8787
- Endpoint chính:
  - `GET /health` — kiểm tra sống.
  - `POST /portfolio/reset` — xoá các CSV trong `in/portfolio/`.
  - `POST /portfolio/upload` — nạp 1 CSV (JSON body: `{name, content}`).
  - `POST /done` — chạy `./broker.sh orders` trên các CSV đã upload và trả về danh sách file đầu vào/đầu ra cùng log thực thi.
- Ghi chú: server chỉ ghi file vào `in/portfolio/` cục bộ và chạy pipeline ngay trong process; không còn bước commit/push hoặc thư mục `runs/`. Policy vẫn được refresh riêng qua `./broker.sh policy` hoặc CI trên nhánh `main`.

Môi trường (env)
- Repo loại bỏ hầu hết biến môi trường “hành vi”. Mặc định chỉ còn:
  - `POLICY_FILE` (tùy chọn): chỉ rõ đường dẫn policy nguồn để merge runtime.
  - `PORT` (server): cổng HTTP, mặc định 8787.
  - Các biến cấu hình CI riêng cho workflow AI (ví dụ `BROKER_CX_GEN_ROUNDS`) không ảnh hưởng hành vi engine.

Policy & cấu hình
- Baseline: `config/policy_default.json` (nguồn sự thật, cố định; không bị workflow ghi đè).
- Overlays: `config/policy_nightly_overrides.json` (do calibrators sinh; commit riêng) và `config/policy_ai_overrides.json` (do tuner sinh; ghi đè trực tiếp overlay AI hiện tại).
- Runtime merge: engine hợp nhất thứ tự baseline → nightly → ai → legacy (nếu có) và ghi policy runtime vào `out/orders/policy_overrides.json`. Khi publish trên nhánh `main`, CI sẽ đồng bộ bản mong muốn vào `config/policy_overrides.json` để phục vụ audit/rollback.

Tài liệu chi tiết
- Kiến trúc, pipeline, nhận diện market regime, thuật toán quyết định lệnh, calibrations, chi tiết merge policy: xem `SYSTEM_DESIGN.md`.

FAQ nhanh / khắc phục sự cố
- Danh mục rỗng hoặc sai đơn vị giá vốn → kiểm tra `in/portfolio/*.csv` (AvgCost tính theo nghìn đồng/cp), xem `out/portfolio_clean.csv` để xác nhận ingest.
- Thiếu dữ liệu lịch sử → chạy lại `./broker.sh orders` (engine tự dựng cache trong `out/data/`).
- Muốn audit thêm lý do/điểm số → xem `out/orders/orders_reasoning.csv` và `out/orders/orders_quality.csv`.

Góp ý & đóng góp
- Mở issue/PR nếu phát hiện lỗi tài liệu/UX; nội dung chuyên sâu xin bổ sung vào `SYSTEM_DESIGN.md` để tránh lặp lại trong README.
