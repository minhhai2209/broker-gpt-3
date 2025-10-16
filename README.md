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
- Không còn biến môi trường để chọn “reasoning effort” hay số vòng phân tích cho Codex: hệ thống mặc định `reasoning_effort=high` và 1 vòng (ổn định, dễ tái lập).
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
- `./broker.sh tune` — chạy calibrators + AI (Codex). Kết quả hợp nhất ghi `out/orders/policy_overrides.json` và được publish sang `config/policy_overrides.json` (tuner copy) để phục vụ audit/rollback.
- `./broker.sh server` — chạy API server cục bộ (Flask) phục vụ extension/ứng dụng (mặc định `PORT=8787`). Server KHÔNG có cron/scheduler nội bộ; việc refresh policy do GitHub Actions hoặc lệnh `./broker.sh policy` thực hiện.
- `python scripts/data_fetching/run_data_jobs.py --group nightly` — chạy toàn bộ nhóm collector chạy đêm (Vietstock fundamentals/events, global factors). Dùng `--dry-run` để kiểm tra config mà không gọi mạng. Log từng job nằm tại `out/logs/data_jobs/<group>/`.
- `python scripts/data_fetching/run_data_jobs.py --job collect_global_factors` (hoặc `collect_vietstock_fundamentals`, `collect_vietstock_events`) — chạy riêng từng collector khi muốn refresh cục bộ mà không ảnh hưởng job khác.

Phân loại collector dữ liệu:
- Nightly (chạy đêm, lâu): cấu hình trong `config/data_jobs.json` nhóm `nightly` gồm Vietstock fundamentals/events và global factors. Các job hỗ trợ chạy song song khi được đánh dấu `allow_parallel=true`, còn Playwright job vẫn chạy tuần tự để tránh tranh chấp trình duyệt.
- Real-time (chạy nhanh trong phiên): nhóm `real_time` hiện tại chỉ bao gồm `ensure_intraday_latest` để cập nhật snapshot phút. Có thể gọi `python scripts/data_fetching/run_data_jobs.py --group real_time` khi cần refresh tức thời (ví dụ sau giờ nghỉ trưa).
- Script tiện ích `scripts/data_fetching/run_collect_all.sh` chỉ là wrapper gọi lần lượt hai nhóm trên; có thể dùng cho cron đơn giản nhưng khuyến nghị dùng trực tiếp `run_data_jobs.py` hoặc `--job` để kiểm soát nhóm/concurrency.
- GitHub Actions tách riêng theo dataset: `Data - Global Factors`, `Data - Vietstock Fundamentals`, `Data - Vietstock Events` (lần lượt chạy `--job` tương ứng sau khi thiết lập Playwright/phụ thuộc). Các workflow này được cron sau giờ đóng cửa HOSE và upload artefact CSV/log để kiểm tra nhanh.

Curated bias (dài hạn, cập nhật thủ công)
- File: `config/policy_curated_overrides.json` (đã được engine merge tự động ở runtime, sau baseline và các overlay khác).
- Chỉ nên khai báo `ticker_bias` (map `{ "TICKER": bias }`, bias ∈ [-0.20..0.20]).
- Nếu không muốn ảnh hưởng logic hạ EXIT→TRIM khi gãy MA50 + RSI thấp, giữ bias < 0.05 (ngưỡng `thresholds.tilt_exit_downgrade_min`).
- TUYỆT ĐỐI không sửa tay `config/policy_overrides.json`; dùng file này cho phần curated lâu dài (bạn có thể cập nhật mỗi tuần).

  

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
  - `PORT` (server): cổng HTTP, mặc định 8787.
  - Các biến cấu hình CI riêng cho workflow AI (ví dụ `BROKER_CX_GEN_ROUNDS`) không ảnh hưởng hành vi engine.

Policy & cấu hình
- Baseline: `config/policy_default.json` (nguồn sự thật, cố định; không bị workflow ghi đè).
- Overlays: `config/policy_nightly_overrides.json` (tuỳ chọn, do calibrators sinh; commit riêng) và unified overlay `config/policy_overrides.json` (hiện hành, do tuner publish).
- ❗ `config/policy_overrides.json` là artefact được sinh tự động sau mỗi lần tune/calibration. Không chỉnh tay, không tạo PR chỉnh tay, và không yêu cầu agent sửa trực tiếp file này — mọi thay đổi sẽ bị lần tune kế tiếp ghi đè và phá vỡ audit trail. Muốn thay đổi hành vi, cập nhật baseline/overlays hợp lệ hoặc chạy lại pipeline.
- Back‑compat: `config/policy_ai_overrides.json` trước đây do tuner sinh; hiện không còn được tạo mặc định, nhưng nếu tồn tại engine vẫn merge.
- Runtime merge: engine hợp nhất baseline → nightly (nếu có) → ai (nếu có) → `config/policy_overrides.json` (legacy) và ghi policy runtime vào `out/orders/policy_overrides.json`. Khi publish trên nhánh `main`, CI giữ `config/policy_overrides.json` làm bản audit.
 - Machine‑generated snapshot: `out/orders/policy_overrides.json` được sinh tự động cho mỗi lần chạy/tune và có trường `"_meta"` như:
   `{ "_meta": { "machine_generated": true, "generated_by": "broker-gpt runtime merge", "generated_at": "..." }, ... }`.
  Không chỉnh tay file này — mọi chỉnh sửa sẽ bị ghi đè. Thay vào đó, cập nhật các overlay trong `config/`.

Slim runtime (policy cleanup)
- Từ 2025‑10‑16, runtime policy được “làm gọn” khi ghi ra `out/orders/policy_overrides.json`:
  - Remove: `calibration`, `thresholds_profiles`, `execution.filter_buy_limit_gt_market`, `execution.fill`.
  - Keep: `thresholds.tp_pct`, `thresholds.sl_pct` luôn hiện diện để hỗ trợ calibrations (engine có thể bỏ qua khi ở chế độ ATR‑dynamic).
  - KEEP: `features.normalization_robust`, `pricing.tc_sell_tax_frac`, `market_bias`, `global_tilts` (được engine dùng runtime).
- Baseline vẫn giữ `tp_pct`/`sl_pct`=0.0 cho tương thích test/schema; chúng được strip ở runtime khi đủ điều kiện.
- Engine luôn kẹp (clamp) BUY Limit xuống Market nếu Limit>Market; không còn filter lệnh tại bước này.

Market filter (VNINDEX)
- Từ 2025-10-16, baseline đặt `market_filter.guard_behavior = "scale_only"`:
  - `scale_only`: không lọc bỏ BUY khi tape yếu thông thường; chỉ co ngân sách bằng các cap (`guard_new_scale_cap`, `atr_soft_scale_cap`) và thang theo `market_score`. Các điều kiện “hard/severe” vẫn đóng băng mua (scale → 0).
  - `pause` (legacy): tạm dừng NEW và hoãn ADD khi guard kích hoạt; chỉ cho phép một số NEW dạng leader-bypass.
- Cách đổi hành vi hợp lệ:
  - Dài hạn: sửa `config/policy_default.json` → `market_filter.guard_behavior` thành `"pause"` hoặc `"scale_only"`, kèm PR cập nhật `SYSTEM_DESIGN.md` (bắt buộc). Không sửa tay `config/policy_overrides.json`.
  - Ngắn hạn (1 phiên): nếu cần “dừng NEW”, tạo patch runtime `out/orders/patch_tune.json` với:
    ```json
    {"meta": {"ttl": "<ISO8601>"}, "exec": {"event_pause_new": 1}}
    ```
    Patch này không thay `guard_behavior` mà chỉ chặn NEW trong phiên, đúng theo engine hỗ trợ runtime.

FAQ (ngắn)
- Vì sao giá đặt trong file lệnh đôi khi bằng giá thị trường? Trong phiên (bao gồm nghỉ trưa), nếu BUY có `LimitPrice` > giá thị trường, hệ thống kẹp về giá thị trường; SELL nếu `LimitPrice` < giá thị trường cũng kẹp về giá thị trường. Quy tắc này chỉ áp ở lớp xuất lệnh, không thay đổi khái niệm “in‑session” ở các module khác.

Tài liệu chi tiết
- Kiến trúc, pipeline, nhận diện market regime, thuật toán quyết định lệnh, calibrations, chi tiết merge policy: xem `SYSTEM_DESIGN.md`.

FAQ nhanh / khắc phục sự cố
- Danh mục rỗng hoặc sai đơn vị giá vốn → kiểm tra `in/portfolio/*.csv` (AvgCost tính theo nghìn đồng/cp), xem `out/portfolio_clean.csv` để xác nhận ingest.
- Thiếu dữ liệu lịch sử → chạy lại `./broker.sh orders` (engine tự dựng cache trong `out/data/`).
- Muốn audit thêm lý do/điểm số → xem `out/orders/orders_reasoning.csv` và `out/orders/orders_quality.csv`.

Góp ý & đóng góp
- Mở issue/PR nếu phát hiện lỗi tài liệu/UX; nội dung chuyên sâu xin bổ sung vào `SYSTEM_DESIGN.md` để tránh lặp lại trong README.
