# Broker GPT — SYSTEM_DESIGN (facts only)

> Bản mô tả kiến trúc/luồng xử lý **dựa trên thực trạng codebase** của repo `broker-gpt-2` (cập nhật tới 2025‑10‑05, VN). Tài liệu này **chỉ nêu sự kiện** (file, hàm, cấu trúc, đầu vào/đầu ra, tham số) theo mã nguồn hiện có; không đánh giá đúng/sai hay so sánh.

---

## 1) Phạm vi & giả định hệ thống

* Ngôn ngữ/Runtime: Python 3.10+.
* Môi trường đích: macOS/Linux/WSL (terminal), có thể chạy qua shell script.
* Phụ thuộc chính: `pandas`, `numpy`, `requests`, `openpyxl`, `pydantic`, `Flask`, `Flask-Cors`, `playwright`, `beautifulsoup4`, `openai` (xem `requirements.txt`).
* Dữ liệu thị trường: sử dụng API VNDirect DChart (HTTP JSON) cho OHLC(EOD/1m) và snapshot nội phiên.
* Dữ liệu cơ bản: có script thu thập từ Vietstock, lưu thành CSV trong `data/`.

---

## 2) Cấu trúc thư mục chính & entrypoints

```
repo/
├─ broker.sh                      # script hợp nhất: orders | tests | policy | server | fundamentals
├─ run_orders.sh                  # helper đơn giản chạy generate_orders.py
├─ requirements.txt
├─ config/
│  ├─ policy_default.json         # baseline policy (commented JSON)
│  └─ policy_overrides.json       # overrides runtime (được sinh/cập nhật bởi công cụ/AI)
├─ data/
│  ├─ industry_map.csv            # map Ticker→Sector
│  └─ fundamentals_vietstock.csv  # (nếu đã crawl) chỉ số cơ bản
├─ in/portfolio/                  # input CSV: Ticker,Quantity,AvgCost (...)
├─ out/                           # toàn bộ artifacts sinh ra khi chạy
│  ├─ data/                       # cache lịch sử OHLC mỗi mã (<T>_daily.csv)
│  ├─ intraday/                   # snapshot nội phiên (latest.csv, *_intraday.csv)
│  ├─ orders/                     # file lệnh và chẩn đoán
│  └─ ... (snapshot.csv, metrics.csv, presets_all.csv, ...)
├─ runs/                          # server API lưu đợt upload để commit
└─ scripts/
   ├─ generate_orders.py          # entrypoint Python cho tác vụ "orders"
   ├─ api/server.py               # Flask API (tùy chọn)
   ├─ engine/                     # mô-đun core: pipeline, config, volatility, risk, calibrators
   ├─ indicators/                 # chỉ báo kỹ thuật nội bộ (MA/RSI/MACD/ATR/Bollinger/Beta)
   ├─ collect_intraday.py         # lấy snapshot nội phiên (VNDirect 1m)
   ├─ fetch_ticker_data.py        # cache OHLC EOD (VNDirect DChart)
   ├─ build_snapshot.py           # tạo out/snapshot.csv
   ├─ build_metrics.py            # tính metrics + session_summary
   ├─ compute_sector_strength.py  # breadth/RSI/ATR theo ngành
   ├─ precompute_indicators.py    # precomputed indicators → out/precomputed_indicators.csv
   ├─ build_presets_all.py        # presets giá (Aggr/Bal/Cons/Break/MR)
   ├─ orders_io.py                # ghi orders_*.csv, tính fill/slippage/limit-lock
   ├─ report_portfolio_pnl.py     # thống kê PnL danh mục (summary/by-sector)
   └─ ai/                         # generate policy_overrides (guardrails, Codex CLI)
```

---

## 3) Luồng end‑to‑end (tác vụ **orders**)

### 3.1. Khởi chạy

* CLI: `./broker.sh` (mặc định `orders`) hoặc `python scripts/generate_orders.py`.
* Script shell chuẩn bị virtualenv, cài `requirements.txt`, rồi chạy entrypoint.

### 3.2. Pipeline dữ liệu (stateless per run)

Pipeline hợp nhất bởi `scripts/engine/pipeline.py`, đảm bảo đủ artifacts trước khi quyết định lệnh. Các bước và đầu ra mặc định:

1. **Ingest danh mục**: đọc các CSV trong `in/portfolio/` → chuẩn hóa mã/đơn vị → `out/portfolio_clean.csv`.
2. **Universe**: lấy danh mục mã từ `data/industry_map.csv`, cộng thêm các chỉ số `VNINDEX, VN30, VN100`, đảm bảo mã trong danh mục nằm trong universe (có thể giới hạn bằng env `BROKER_UNI_LIMIT`).
3. **Lịch sử OHLC**: gọi VNDirect DChart, cache mỗi mã vào `out/data/<Ticker>_daily.csv`; hợp nhất thành `out/prices_history.csv` (có cột `Date,Ticker,Open,High,Low,Close,Volume,t`).
4. **Intraday snapshot**: lấy 1‑phút gần nhất cho universe vào `out/intraday/latest.csv` (best‑effort; có cơ chế fallback cache).
5. **Snapshot & Metrics**:

   * `out/snapshot.csv`: ảnh chụp giá hiện tại (ghép intraday/daily/override giá).
   * `out/metrics.csv`: RSI14, ATR14%, ADTV20k, Beta60D, MACD‑Hist, Mom(12‑1/6‑1), TickSizeHOSE…; kèm `Sector` (từ industry_map) và thông tin phiên.
   * `out/session_summary.csv`: `SessionPhase, InVNSession, VNIndex, IndexChangePct, Advancers/Decliners, TotalValue`.
   * (nếu có) ghép dữ liệu cơ bản từ `data/fundamentals_vietstock.csv` → `out/fundamentals_snapshot.csv`.
6. **Sector strength**: breadth theo MA20/50/200, ATR%, RSI trung bình theo ngành → `out/sector_strength.csv`.
7. **Precomputed indicators**: MA/ATR/đỉnh‑đáy, … → `out/precomputed_indicators.csv`.
8. **Presets giá** (per ticker): dải band ngày ±7%, tính `*_Buy1/2/Sell1/2(_Tick)` cho 5 preset `Cons/Bal/Aggr/Break/MR` → `out/presets_all.csv`.
9. **PnL**: thống kê nhanh danh mục → `out/portfolio_pnl_summary.csv`, `out/portfolio_pnl_by_sector.csv`.

### 3.3. Policy & tuning tại runtime

* Baseline: `config/policy_default.json` (định nghĩa `weights`, `thresholds`, `pricing`, `sizing`, `execution`, `market_filter`, `regime_model`, `orders_ui`...).
* Overrides: `config/policy_overrides.json` (chỉ tập khóa whitelisted) → hợp nhất có chọn lọc → ghi bản **runtime** vào `out/orders/policy_overrides.json`.
* Tập khóa **được phép override** (ở cấp map): `buy_budget_frac`, `add_max`, `new_max`, `sector_bias`, `ticker_bias`, `thresholds`, `sizing`, `pricing`, `orders_ui`, `market_filter`, `regime_model`.
* Bộ sinh overrides (tùy chọn): `scripts/ai/generate_policy_overrides.py` sử dụng Codex CLI + guardrails để sinh/cập nhật `config/policy_overrides.json`.

### 3.4. Nhận diện **Market Regime** (phiên hiện tại)

* Đọc `out/prices_history.csv` (series VNINDEX) để tính: Garman‑Klass sigma (vol), percentile, momentum 63D, drawdown, trend_strength (MA200/MA50), và bucket TTL (low/medium/high).
* Các trường trong đối tượng `MarketRegime` gồm: phase, in_session, index_change_pct, breadth_hint, risk_on, buy_budget_frac, add_max/new_max, weights, thresholds, sector/ticker_bias, pricing/sizing/execution, các percentile/diagnostics (vol, momentum, drawdown…), TTL bucket/state và neutral‑adaptive metadata.

### 3.5. Quyết định hành động & xếp hạng

* Chấm điểm theo `weights` (trend/momentum/liquidity/beta/sector/fundamentals…) và áp điều kiện trong `thresholds` (`base_add`, `base_new`, `trim_th`, `q_add`, `q_new`, `min_liq_norm`, `near_ceiling_pct`, `tp/sl` dạng phần trăm hoặc ATR‑based, v.v.).
* Hành động sinh ra gồm các nhãn như: `add`, `new`, `trim`, `take_profit`, `exit` (các lệnh SELL có thể gắn thêm meta `STOP_FINAL`, TTL override…).
* Bộ lọc microstructure: chặn/giảm ưu tiên khi giá **gần trần** (near‑ceiling) theo `thresholds.near_ceiling_pct`.
* Chế độ **neutral‑adaptive** (tùy cấu hình): có thể sinh `partial entry`, giới hạn `add_max`, theo dõi danh sách neutral_*.

### 3.6. Định giá lệnh (limit price) & tick size

* Ưu tiên preset theo chế độ thị trường: `risk_on_buy/sell`, `risk_off_buy/sell` (ví dụ BUY ưu tiên Aggr→Bal→Cons…).
* Fallback theo ATR nếu preset không khả dụng; mọi giá được **round** theo bước giá HOSE (0.01/0.05/0.10 nghìn).
* Ràng buộc dải ngày (BandFloor/BandCeiling) để tránh đặt ngoài biên.

### 3.7. Ghi đầu ra

* **File để nhập lệnh**: `out/orders/orders_final.csv` (4 cột cố định: `Ticker,Side,Quantity,LimitPrice`) — sắp xếp BUY trước, sau đó theo ưu tiên nội bộ.
* **Bảng chẩn đoán**:

  * `out/orders/orders_quality.csv`: ước lượng `FillProb`, `FillRateExp`, `SlipBps/Pct`, `LimitLock`, Priority (nội bộ)…
  * `out/orders/orders_reasoning.csv`: Action, Score và các feature chính (above_ma20/50, rsi, macdh_pos, liq_norm, atr_pct, pnl_pct).
  * `out/orders/orders_print.txt`: bản in tóm tắt lệnh và tổng tiền mua/bán.
  * `out/orders/orders_analysis.txt`: phân tích bộ lọc, hint khung thời gian thực thi, v.v.
  * `out/orders/regime_components.json`: snapshot các chỉ số chế độ thị trường/diagnostics.
* **Các file khác (tùy)**: `orders_watchlist.csv`, `trade_suggestions.txt`, `portfolio_evaluation.(txt|csv)`.

---

## 4) Chính sách (Policy) & Calibrations

### 4.1. Policy baseline & overrides

* Baseline lưu toàn bộ tham số chiến lược (commented JSON).
* Overrides merge theo whitelist; kết quả runtime ghi vào `out/orders/policy_overrides.json` và được toàn bộ luồng sử dụng.

### 4.2. Guardrails cho overrides (khi sinh bằng AI)

* Ràng buộc biên độ/TTL cho `buy_budget_frac`, `add_max`, `new_max`; **bias** theo sector/ticker có TTL/decay; hỗ trợ `news_risk_tilt` (map sang budget/slots) và ghi audit JSONL/CSV trong `out/orders/`.

### 4.3. Calibrators hiện diện trong codebase

* **TTL minutes**: script `scripts/engine/calibrate_ttl_minutes.py` đọc VNINDEX σ (Garman‑Klass) → bucket {low, medium, high} và cập nhật `orders_ui.ttl_minutes` + metadata (`ttl_bucket_*`).
* **Mean‑variance sizing**: module `scripts/engine/mean_variance_calibrator.py` hiệu chỉnh nhanh các tham số `risk_alpha`, `cov_reg`, `bl_alpha_scale` bằng walk‑forward trên cửa sổ lịch sử; dùng cùng bộ giải `compute_cov_matrix` / `compute_expected_returns` / `solve_mean_variance_weights` / `compute_risk_parity_weights` trong `portfolio_risk.py`.
* **Cờ env CALIBRATE_***: trong test/luồng runtime có các biến môi trường để bật/tắt nhóm calibrations (ví dụ `CALIBRATE_MARKET_FILTER`, `CALIBRATE_LIQUIDITY`, `CALIBRATE_THRESHOLDS_TOPK`, `CALIBRATE_SIZING_TAU`, …).

---

## 5) Server API (tùy chọn)

* Khởi chạy: `./broker.sh server` (mặc định `PORT=8787`).
* Endpoint:

  * `GET /health` → `{status: ok, ts: ...}`
  * `POST /portfolio/reset` → xóa file trong `in/portfolio/` và reset phiên tải lên.
  * `POST /portfolio/upload` (JSON `{name, content}`) → ghi file vào `in/portfolio/` **và** `runs/<stamp>/portfolio/`.
  * `POST /done` → commit các CSV trong `runs/<stamp>/portfolio/` (git add/commit/push) rồi trả về danh sách đã commit cùng trạng thái **PolicyScheduler** (nếu bật).
* **PolicyScheduler**: nếu `BROKER_POLICY_AUTORUN=1`, tiến trình nền lập lịch chạy `broker.sh policy` theo `BROKER_POLICY_TIMES` (chuỗi `HH:MM`), với `lead` phút trước slot và `BROKER_POLICY_TZ` (mặc định `Asia/Ho_Chi_Minh`).

---

## 6) Tham số & biến môi trường đáng chú ý

* **BROKER_UNI_LIMIT**: giới hạn số mã trong universe khi phát triển/test.
* **POLICY_FILE**: trỏ tới file policy để merge (nếu cần); mặc định dùng baseline+overrides.
* **BROKER_COVERAGE**: bật coverage cho `./broker.sh tests`.
* **BROKER_POLICY_* / BROKER_RUNS_TZ / BROKER_ARCHIVE_TZ**: cấu hình server/scheduler và timezone.
* **CALIBRATE_***: bật/tắt nhóm calibrations khi chạy engine/tests.

---

## 7) Chỉ báo kỹ thuật & helper nội bộ

* `scripts/indicators/`: MA, RSI(Wilder), MACD‑Hist, ATR(Wilder), Bollinger bands, Beta rolling.
* `scripts/utils.py`: `hose_tick_size` (0.01/0.05/0.10 nghìn), `round_to_tick`, `clip_to_band`, `detect_session_phase_now_vn`.

---

## 8) Kiểm thử (tests)

* Có tests mạng cho VNDirect (`test_network_fetch_ticker_data.py`, `test_network_collect_intraday.py`, `test_network_build_snapshot.py`).
* Tests hành vi order engine: near‑ceiling guard, stop/take‑profit/trim, TTL overrides, file diagnostics (`orders_analysis.txt`, `regime_components.json`).
* Tests sizing/risk: covariance (Ledoit‑Wolf + fallback), expected returns (CAPM + score view), risk‑parity solver.
* Tests luồng sinh policy_overrides qua Codex CLI + guardrails (file‑flow, END/CONTINUE, audit/TTL/bounds).

---

## 9) Tóm tắt đầu vào/đầu ra chuẩn

**Đầu vào**

* `in/portfolio/*.csv` với schema tối thiểu `Ticker,Quantity,AvgCost` (AvgCost tính **nghìn VND/cp**).
* `data/industry_map.csv` (bắt buộc để gán Sector & validate).
* (tùy chọn) `data/fundamentals_vietstock.csv`, `config/price_overrides.csv`.

**Đầu ra chính trong `out/`**

* `prices_history.csv`, `intraday/latest.csv`, `snapshot.csv`, `metrics.csv`, `sector_strength.csv`, `precomputed_indicators.csv`, `presets_all.csv`, `session_summary.csv`.
* `orders/orders_final.csv`, `orders_quality.csv`, `orders_reasoning.csv`, `orders_print.txt`, `orders_analysis.txt`, `regime_components.json`.
* `portfolio_pnl_summary.csv`, `portfolio_pnl_by_sector.csv`.

---

## 10) Ghi chú về đơn vị/định dạng

* Giá trị tiền **tính theo nghìn VND** trong phần lớn bảng (ví dụ `LimitPrice` trong orders_final.csv là **nghìn/cp**).
* Bước giá HOSE và lô chẵn 100 được áp dụng trong tính toán tick & lượng.
* TTL (phút) có thể được **calibrate** theo bucket vol thấp/vừa/cao và ghi vào `orders_ui.ttl_minutes` trong policy runtime.

---

*Hết.*
