# Prompt gợi ý cho ChatGPT

Sao chép toàn bộ nội dung dưới đây vào ChatGPT (hoặc công cụ tương tự). Prompt này yêu cầu đọc các CSV đầu ra và tự giải thích preset dựa trên mô tả đi kèm.

```
Hãy tra cứu tin tức hiện tại và đọc các file sau (đường dẫn tương đối) để đề xuất bộ lệnh cho phiên VNINDEX tới, chỉ dùng lệnh LO:
- out/market/technical_snapshot.csv — ảnh chụp kỹ thuật theo mã (giá hiện tại, thay đổi %, SMA/EMA/RSI/ATR, MACD, Z‑score, returns, ADV, 52w range)
- out/presets/balanced.csv — mức bậc mua/bán theo preset 'balanced'.
- out/presets/momentum.csv — mức bậc mua/bán theo preset 'momentum'.
- data/portfolios/alpha.csv — danh mục hiện tại (Ticker, Quantity, AvgPrice)
- out/portfolios/alpha_positions.csv — PnL theo mã, MarketValue/CostBasis/Unrealized
- out/portfolios/alpha_sector.csv — tổng hợp PnL theo ngành
- data/order_history/alpha_fills.csv — các lệnh đã khớp trong hôm nay

Yêu cầu khi ra quyết định:
- Ưu tiên preset phù hợp theo mô tả sau (đọc kỹ trước khi quyết định):
  - balanced: Thiết lập mặc định cân bằng giữa tích lũy và chốt lời. Mua từng phần khi điều chỉnh vừa phải; chốt lời theo bậc tăng dần.
  - momentum: Mua theo đà, chốt lời nhanh. Ưu tiên mã đang có động lượng; dừng lỗ sớm nếu đà gãy.
  - Nếu có preset khác ngoài hai preset trên, dùng các mức Buy_i/Sell_i và giải thích logic từ tên preset.
- Chỉ dùng lệnh LO, tính LimitPrice theo nghìn đồng.
- Tôn trọng quy tắc HOSE (bước giá, lô chẵn, biên độ) ở cuối prompt.
- Không vượt quá khối lượng hợp lệ theo lô chẵn; có thể đề xuất nhiều lệnh nhỏ thay vì một lệnh lớn.

Xuất kết quả duy nhất dưới dạng CSV với header: `Ticker,Side,Quantity,LimitPrice`.
- `Side` là `BUY` hoặc `SELL`.
- `LimitPrice` ghi theo đơn vị nghìn đồng.

Quy tắc HOSE (áp dụng khi tính giá/khối lượng hợp lệ):
- Đơn vị yết giá (tick) cổ phiếu/CCQ đóng: <10.000 VND → 10; 10.000–49.950 → 50; ≥50.000 → 100. ETF/CW: 10.
- Lô chẵn: bội số 100; tối đa 500.000 cổ/lệnh. Biên độ HOSE: ±7% so với tham chiếu.
- Làm tròn giá trần/sàn theo quy chế: trần làm tròn xuống, sàn làm tròn lên theo đúng tick.

Gợi ý kiểm tra nhanh (giá báo theo nghìn đồng):
- `p_vnd = round(LimitPrice * 1000)`; chọn `tick` theo mức `p_vnd`.
- Hợp lệ khi `(p_vnd % tick == 0)` và `Quantity % 100 == 0`.
```

Ghi chú:
- Mọi preset có thể có cột `PresetDescription`. Hãy dựa vào cột này để hiểu ý nghĩa và cách vận dụng preset.
- File/đường dẫn ví dụ dùng profile `alpha`; thay bằng profile của bạn nếu khác.
