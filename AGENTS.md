# Broker GPT — Agent Notes

## Way of Work
- Tuân thủ lý thuyết đã được thừa nhận; không tự chế thuật toán khi thiếu bằng chứng học thuật/thực nghiệm.
- Đánh giá thị trường dựa trên dữ liệu chuẩn; thiếu dữ liệu thì dừng pipeline và yêu cầu bổ sung.
- Fail fast, không nuốt lỗi; tránh try/except rộng và luôn raise với thông điệp rõ ràng, có log.
  - CI/Automation policy: pipeline phải FAIL (exit != 0) khi thiếu cấu hình/bí mật bắt buộc. Không dùng cảnh báo để tiếp tục chạy (không "proceed without ..."). Ví dụ: thiếu `.codex/config.toml` hoặc secret → dừng ngay với thông báo rõ ràng.
- Xác thực dữ liệu vào/ra; kiểm tra schema/cột/độ dài; numeric null phải bị chặn thay vì đoán.
- Không ẩn default trong engine; giá trị mặc định đặt tại schema/baseline và tài liệu hoá rõ ràng.
- Giữ contract rõ giữa module; không trả về DataFrame rỗng hoặc `{}` khi pipeline lỗi; surface nguyên nhân cho caller.
- Calibrate dựa trên dữ liệu khách quan; mọi chủ quan chỉ áp khi đã chốt trong policy và được ghi chú.
- Nhật ký minh bạch; ghi diagnostics để người dùng hiểu quyết định, tránh “blackbox”.
- Mặc định ổn định — không dùng env để tắt/bật; hành vi cố định bằng defaults hợp lý và policy.
- Schema & test đi cùng thay đổi: khi chỉnh schema/policy, cập nhật baseline, whitelist overrides liên quan và test/fixtures tương ứng (giữ tests xanh).
- Test/coverage trước bàn giao: chạy `./broker.sh tests` (có thể bật coverage) sau mỗi thay đổi quan trọng.
