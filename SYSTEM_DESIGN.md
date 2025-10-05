Kiến trúc Hệ thống Broker GPT

Tổng Quan Hệ Thống

Broker GPT là một hệ thống đưa ra đề xuất giao dịch cổ phiếu theo danh mục hiện có của người dùng, được thiết kế theo mô hình pipeline xử lý dữ liệu kết hợp với một bộ máy ra quyết định lệnh (order engine) hoạt động dựa trên cấu hình chiến lược (policy) và tín hiệu thị trường. Hệ thống hướng đến tính stateless – mỗi lần chạy engine là độc lập, không giữ trạng thái trong bộ nhớ giữa các phiên. Tuy nhiên, engine vẫn lưu lại một số trạng thái phiên trước dưới dạng file để sử dụng cho lần chạy sau (ví dụ: file out/orders/position_state.csv ghi nhận mã nào đã chốt lời/dừng lỗ một phần ở phiên trước)【8†L8-L11】【34†L136-L140】. Kết quả đầu ra mỗi lần chạy là bộ file lệnh và báo cáo trong thư mục out/ hoặc thư mục lưu trữ theo timestamp, có thể dùng để nhập lệnh giao dịch thực tế hoặc phân tích.

Về triển khai, Broker GPT chủ yếu viết bằng Python (yêu cầu Python 3.10+), chạy tốt trên môi trường máy cá nhân (macOS/Linux/WSL). Hệ thống có hai cách sử dụng chính: (1) Chạy qua CLI với script broker.sh cho phép thực thi pipeline tạo lệnh hoặc thực hiện các tác vụ hỗ trợ (test, cập nhật dữ liệu, v.v.), và (2) Chạy qua một API server Flask (chế độ broker.sh server) để phục vụ tích hợp với frontend (ví dụ extension trình duyệt). Hệ thống cũng tích hợp với quy trình CI/CD (GitHub Actions) để thực thi pipeline một cách tự động và phi đồng bộ khi nhận được yêu cầu từ server. Dưới đây là các thành phần chính của hệ thống và cách chúng tương tác:

- Pipeline xử lý dữ liệu: Thu thập và xử lý tất cả dữ liệu đầu vào cần thiết (danh mục, dữ liệu thị trường quá khứ và hiện tại, chỉ số ngành, yếu tố bổ trợ) để chuẩn bị cho bước quyết định lệnh【8†L4-L8】.
- Cấu hình chiến lược (Policy): Bộ tham số chiến lược điều chỉnh hành vi của engine, gồm một cấu hình mặc định cố định và phần cấu hình linh hoạt (override) có thể thay đổi theo ngày. Hệ thống hợp nhất hai phần này để có policy runtime cho mỗi phiên chạy【8†L5-L7】.
- Bộ máy ra quyết định lệnh (Order Engine): Sử dụng dữ liệu từ pipeline và policy đã hợp nhất để nhận diện chế độ thị trường hiện tại và tính toán hành động mua/bán/giữ cho từng mã cổ phiếu trong hoặc ngoài danh mục. Thành phần này chính là “bộ não” của hệ thống, quyết định tạo ra lệnh gì với số lượng bao nhiêu và giá nào【8†L6-L8】.
- Tương tác ngoài & tích hợp frontend: Engine lấy dữ liệu từ các nguồn bên ngoài (API, web) khi cần – ví dụ gọi API VNDirect để lấy giá, dùng Playwright để thu thập dữ liệu cơ bản từ Vietstock【8†L7-L9】. Đồng thời, hệ thống cung cấp một API server (Flask) phục vụ cho extension hoặc ứng dụng bên ngoài: nhận danh mục upload, kích hoạt engine chạy, rồi trả về kết quả lệnh. Phần server này giúp tích hợp Broker GPT vào giao diện người dùng một cách thuận tiện mà vẫn tận dụng core engine chạy nền.
- Cơ chế hàng đợi & thực thi bất đồng bộ: Do việc tính toán của engine có thể mất thời gian, server Flask không chạy thẳng engine trong request mà sử dụng một cơ chế hàng đợi thông qua Git để kích hoạt xử lý nền. Cụ thể, mỗi yêu cầu chạy lệnh từ frontend sẽ được server commit dữ liệu danh mục vào repo (thư mục runs/) và đẩy (push) lên remote, từ đó kích hoạt GitHub Actions runner đóng vai trò worker để thực thi pipeline và trả kết quả về lại repo【13†L399-L407】【27†L2-L10】. Cách tiếp cận này biến GitHub Actions thành một hàng đợi xử lý lệnh phi đồng bộ (mỗi commit tương ứng một “job” chạy engine). Kết quả tính toán được commit vào thư mục runs/<timestamp>/ tương ứng.

Các phần sau sẽ đi chi tiết vào từng thành phần, mô tả cách chúng hoạt động và gắn kết với nhau trong kiến trúc tổng thể.

Pipeline Xử Lý Dữ Liệu (Data Pipeline)

Pipeline dữ liệu chịu trách nhiệm chuẩn bị toàn bộ dữ liệu đầu vào cần thiết cho engine trước khi đưa ra quyết định giao dịch. Mã nguồn triển khai chủ yếu nằm trong scripts/engine/pipeline.py (hàm ensure_pipeline_artifacts()), cùng các module hỗ trợ như scripts/build_snapshot.py, scripts/build_metrics.py, scripts/precompute_indicators.py, v.v. Pipeline luôn được gọi tự động khi chạy engine (từ script generate_orders.py), do đó người dùng chỉ cần cung cấp đầu vào và chạy lệnh, không cần chạy pipeline thủ công【28†L26-L29】. Các bước chính của pipeline như sau:

1) Nạp và làm sạch danh mục hiện có: Engine đọc danh mục từ các file CSV trong thư mục in/portfolio/ (có thể có một hoặc nhiều file, hệ thống sẽ hợp nhất) với schema cơ bản gồm các cột như Ticker, Quantity, AvgCost. Mã cổ phiếu được chuẩn hóa (viết hoa, bỏ ký tự đặc biệt như *), đơn vị giá vốn quy đổi về nghìn đồng/cổ phiếu. Kết quả là một DataFrame portfolio_df sạch, đồng thời ghi ra file out/portfolio_clean.csv để audit【8†L15-L18】. Nếu danh mục trống hoặc không chứa mã hợp lệ, engine sẽ dừng và báo lỗi – đảm bảo có ít nhất một mã thuộc sàn HOSE làm vũ trụ phân tích.

2) Xác định vũ trụ mã phân tích (universe): Dựa trên danh mục của người dùng và dữ liệu sẵn có, engine xây dựng danh sách mã cần phân tích. Mặc định, engine tải danh sách tất cả mã và ngành từ file tĩnh data/industry_map.csv (coi như vũ trụ cơ bản). Sau đó, engine đảm bảo mọi mã trong danh mục đều có trong vũ trụ (nếu chưa có thì thêm), đồng thời bổ sung các chỉ số thị trường chính như VNINDEX, VN30, VN100. Kết quả thu được là danh sách uni các ticker cần lấy dữ liệu【8†L17-L19】. (Hệ thống có biến môi trường BROKER_UNI_LIMIT để giới hạn số mã cho mục đích phát triển/test, nhưng mặc định không giới hạn).

3) Tải dữ liệu giá lịch sử (historical prices): Sử dụng danh sách uni trên, engine gọi hàm ensure_and_load_history_df(uni, ...) (định nghĩa trong scripts/fetch_ticker_data.py) để thu thập dữ liệu giá daily (OHLC) khoảng ~700 ngày gần nhất cho tất cả mã trong vũ trụ【8†L18-L20】. Việc tải được tối ưu nhờ cơ chế cache: hàm trên kiểm tra trong thư mục out/data/ xem đã có file CSV lịch sử cho từng mã hay chưa; nếu có, chỉ tải bổ sung phần thiếu (những ngày mới). Nguồn dữ liệu giá lịch sử là API VNDirect DChart được gọi qua HTTP requests để lấy dữ liệu theo khoảng thời gian cần thiết【8†L18-L20】. Mỗi mã được lưu cache vào một file CSV riêng (tên như <Ticker>_daily.csv trong out/data/). Kết quả bước này là DataFrame prices_history_df chứa toàn bộ dữ liệu giá quá khứ của các mã vũ trụ (khoảng >=400 phiên gần nhất). Pipeline cũng ghép tất cả thành file hợp nhất out/prices_history.csv để thuận tiện kiểm tra.

4) Cập nhật dữ liệu giá nội phiên (intraday): Nếu thời điểm chạy engine đang trong giờ giao dịch (xác định theo múi giờ Việt Nam và lịch HOSE), engine sẽ lấy snapshot giá mới nhất trong phiên cho các mã. Hàm ensure_intraday_latest(uni, ...) (trong module scripts/collect_intraday.py) được gọi để thu thập giá hiện tại của các mã vũ trụ【8†L18-L21】. Việc này có thể thực hiện qua API hoặc web tuỳ nguồn dữ liệu (ví dụ cũng từ VNDirect hoặc nguồn web khác). Kết quả được lưu vào out/intraday/latest.csv. Nếu chạy ngoài giờ giao dịch, bước này có thể bị bỏ qua hoặc dùng dữ liệu chốt phiên trước. (Hệ thống xác định phiên sáng/chiều dựa trên múi giờ Asia/Ho_Chi_Minh và lịch thị trường để biết khi nào cần intraday hay không).

5) Tạo snapshot giá hiện tại và tính toán chỉ số phiên: Từ dữ liệu vừa có, engine xây dựng snapshot_df – bảng giá hiện tại của toàn bộ mã (giá cuối cùng hoặc giá mới nhất, cùng các thông tin cơ bản khác cho mỗi mã). Thao tác này do hàm build_snapshot_df(portfolio_df, ...) trong scripts/build_snapshot.py thực hiện【8†L20-L23】. Đồng thời, engine tính toán một loạt metrics kỹ thuật cho từng mã qua hàm build_metrics_df(...) trong scripts/build_metrics.py. Các metrics bao gồm: biến động 20 phiên, thanh khoản trung bình 20 phiên (ADTV), RSI14, ATR%, hệ số beta 60 ngày, sức mạnh giá tương đối, v.v. Bên cạnh đó, engine tổng hợp thông tin phiên thị trường (session summary) như: VN-Index thay đổi bao nhiêu %, tỷ lệ cổ phiếu tăng/giảm (breadth) so với MA50, trạng thái thị trường hiện tại (đang trong phiên hay ngoài giờ), v.v. Kết quả gồm DataFrame metrics_df cho từng mã và session_summary_df cho thị trường chung. Ngoài ra, engine gắn nhãn ngành cho từng mã (theo map ngành từ industry_map) để phục vụ tính toán phân bổ sau này【8†L20-L23】. Nếu có file dữ liệu cơ bản (ví dụ data/fundamentals_vietstock.csv thu thập từ Vietstock), engine sẽ merge vào metrics các chỉ số như P/E, ROE, Earnings Yield cho từng mã【8†L20-L23】. Tất cả dữ liệu snapshot, metrics, session summary (và fundamentals nếu có) được ghi ra các file CSV tương ứng trong thư mục out/ (snapshot.csv, metrics.csv, session_summary.csv, fundamentals_snapshot.csv…).

6) Tính toán sức mạnh ngành và chỉ báo kỹ thuật lịch sử: Engine nhóm dữ liệu theo ngành để tính Sector Strength – đại diện mức tăng/giảm tương đối của mỗi ngành so với toàn thị trường. Kết quả là DataFrame sector_strength_df, ghi ra out/sector_strength.csv. Song song, engine tiền tính một số indicators kỹ thuật lịch sử cho từng mã để phục vụ tính điểm tín hiệu sau này. Hàm precompute_indicators_from_history_df(...) trong scripts/precompute_indicators.py sử dụng chuỗi giá lịch sử để tính các thông số như đường MA, mức đỉnh/đáy 52 tuần, ATR%, v.v., lưu kết quả vào out/precomputed_indicators.csv【28†L22-L24】.

7) Tích hợp yếu tố vĩ mô (tùy chọn): Nếu người dùng cung cấp file data/global_factors.csv chứa các chỉ số vĩ mô (ví dụ: S&P 500, chỉ số USD, chỉ số EPU – Economic Policy Uncertainty), pipeline sẽ bổ sung các feature vĩ mô vào dữ liệu phiên. Ví dụ: tính phần trăm phân vị của EPU (so với lịch sử), mức giảm từ đỉnh của S&P500 (drawdown), phân vị sức mạnh USD, v.v. Những thông tin này được thêm vào metrics_df hoặc session_summary_df, đồng thời engine kiểm tra các ngưỡng market guardrails tương ứng trong policy (ví dụ us_epu_soft_pct, spx_drawdown_hard_pct trong cấu hình market_filter). Nếu thiếu dữ liệu hoặc không có cấu hình, bước này bị bỏ qua【28†L23-L25】. (Các yếu tố vĩ mô này chủ yếu phục vụ xác định chế độ thị trường và kích hoạt những bộ lọc thị trường sẽ đề cập sau).

8) Xây dựng Preset theo chế độ thị trường: Broker GPT có khái niệm preset tham số cho các kịch bản thị trường khác nhau (ví dụ “bảo thủ”, “cân bằng”, “tấn công”). Pipeline tạo ra bảng presets cho mỗi mã thông qua hàm build_presets_all_df(...) trong scripts/build_presets_all.py. Hàm này kết hợp dữ liệu snapshot, lịch sử giá và trạng thái phiên để gán các thông số preset phù hợp cho mã ở các chế độ thị trường (như mức giá mua/bán ưu tiên khi thị trường risk-on vs risk-off). Kết quả presets_df ghi ra out/presets_all.csv【28†L24-L26】. Preset được dùng sau này trong tính toán giá đặt lệnh (pricing) tùy chế độ.

9) Tính toán P&L tạm tính của danh mục: Cuối pipeline, engine tính nhanh lãi/lỗ danh mục hiện tại dựa trên giá hiện tại so với giá vốn. Điều này tạo báo cáo tham khảo về hiệu suất danh mục. Kết quả gồm các file out/portfolio_pnl_summary.csv (tổng P&L tuyệt đối và % cho cả danh mục) và out/portfolio_pnl_by_sector.csv (P&L phân theo ngành)【28†L25-L27】. Bước này chỉ nhằm mục đích thông tin, không ảnh hưởng đến quyết định lệnh.

Hoàn thành các bước trên, pipeline đảm bảo mọi dữ liệu cần thiết đã sẵn sàng trong thư mục out/ để audit và sử dụng cho việc ra quyết định. Các file trung gian (danh mục sạch, lịch sử giá, snapshot, metrics, sector_strength, presets, P&L…) cung cấp khả năng kiểm tra lại từng bước nếu cần. Lưu ý, nếu chạy thông qua script broker.sh orders, pipeline sẽ tự động được gọi; người dùng không cần thao tác riêng biệt【28†L26-L29】. Miễn là danh mục đầu vào có sẵn và kết nối mạng để tải dữ liệu, pipeline sẽ xử lý toàn bộ trước khi chuyển sang phần tính toán lệnh.

Cấu Hình Chiến Lược (Policy) và Điều Chỉnh Tham Số

Policy của Broker GPT là tập hợp các tham số chiến lược chi phối cách engine đánh giá tín hiệu và quản trị rủi ro. Hệ thống tổ chức cấu hình này thành hai tầng: mặc định (baseline) và điều chỉnh hằng ngày (overrides). Mục đích là giữ ổn định chiến lược lõi nhưng vẫn cho phép tinh chỉnh linh hoạt theo diễn biến thị trường ngắn hạn. 

- Policy mặc định: Được định nghĩa trong file config/policy_default.json. Đây là nguồn sự thật chứa toàn bộ tham số chiến lược cơ bản, kết tinh từ nghiên cứu dài hạn. Ví dụ, trong policy mặc định có: trọng số mô hình điểm cho các yếu tố (xu hướng, động lượng, thanh khoản, beta, v.v.), các ngưỡng kỹ thuật như base_add, base_new (điểm tối thiểu để mua bổ sung/mua mới), ngưỡng trim_th để cắt giảm vị thế khi điểm yếu, các ngưỡng chốt lời (tp_pct) và cắt lỗ (sl_pct), tham số vi mô về khớp lệnh (bước giá HOSE, lô 100, phí giao dịch), giới hạn rủi ro (tỷ trọng tối đa cho một mã, một ngành), v.v. Hầu hết các giá trị trong policy_default là cố định, chỉ thay đổi khi điều chỉnh chiến lược lớn hoặc sau quá trình backtest dài hạn【28†L32-L40】.

- Policy overrides (điều chỉnh hàng ngày): Được định nghĩa trong file config/policy_overrides.json. File này cho phép ghi đè một số ít tham số so với mặc định, nhằm tùy biến chiến lược cho phù hợp với hoàn cảnh thị trường hoặc tin tức mỗi ngày. Mục tiêu của override là giúp hệ thống phản ứng nhanh với các yếu tố khó định lượng (tâm lý thị trường, sự kiện bất ngờ) mà không cần thay đổi toàn bộ cấu hình. Chỉ một tập giới hạn các khóa được phép override, các phần còn lại của policy luôn tuân theo mặc định để đảm bảo ổn định. Cụ thể, các khóa cho phép override bao gồm【28†L33-L40】:

  - buy_budget_frac – Tỷ lệ ngân sách dành cho lệnh mua trên tổng NAV. Tham số này điều chỉnh mức độ “risk-on/risk-off” chung (mua nhiều hay hạn chế mua).
  - add_max và new_max – Số lệnh mua tối đa cho danh mục hiện tại (add_max) và cho mã mới (new_max) mà engine được phép đề xuất trong phiên. Điều chỉnh hai giá trị này sẽ kiểm soát nhịp độ giải ngân (mua bổ sung thêm bao nhiêu mã đang có, và mua mới tối đa bao nhiêu mã mới)【28†L34-L37】.
  - sector_bias và ticker_bias – Hệ số thiên lệch (bias) cho một số ngành hoặc mã cụ thể. Mỗi bias là một giá trị trong khoảng [-0.20 .. +0.20] áp dụng lên điểm tín hiệu của ngành/mã đó. Bias dương nghĩa là ưu tiên (tăng điểm để khuyến khích mua), bias âm nghĩa là thận trọng (giảm điểm để hạn chế mua). Tham số này cho phép phản ánh tin tức đặc biệt: ví dụ tin tốt về một ngành -> tăng nhẹ điểm của ngành đó, tin xấu -> giảm điểm ngành đó【28†L35-L38】. Mỗi bias thường đi kèm một giải thích (rationale) và có thể tự động giảm dần hiệu lực sau vài phiên nếu không được cập nhật thêm.
  - news_risk_tilt (tuỳ chọn) – Thang đo tâm lý thị trường chung từ tin tức, giá trị [-1 .. +1]. Giá trị âm nghĩa là tâm lý “risk-off” (lo ngại), dương là “risk-on” (lạc quan). Nếu cung cấp, hệ thống sẽ ánh xạ tilt này thành điều chỉnh tự động cho các tham số như buy_budget_frac, add_max, new_max (theo guardrails nội bộ). Tham số này không được lưu giữ dài hạn mà chỉ dùng như một input phụ trợ tạm thời.
  - rationale – Chuỗi mô tả ngắn gọn lý do cho các điều chỉnh trên (bắt buộc phải có mỗi khi override để phục vụ audit).

Các khóa trên là duy nhất những phần policy có thể thay đổi hàng ngày bởi AI hoặc người vận hành; toàn bộ phần còn lại của policy (như ngưỡng tính toán từ dữ liệu, tham số cố định) sẽ không bị ảnh hưởng bởi override để tránh nhiễu loạn chiến lược【28†L33-L40】【28†L41-L47】. Cách tiếp cận này đảm bảo hệ thống giữ được tính ổn định dài hạn nhưng vẫn có độ nhạy cần thiết trước diễn biến ngắn hạn.

Quy trình hợp nhất cấu hình: Mỗi lần engine chạy, nó sẽ hợp nhất policy mặc định và override để tạo ra cấu hình chiến lược runtime cho phiên đó. Hàm ensure_policy_override_file() trong scripts/engine/config_io.py đảm nhiệm việc này. Cơ chế như sau:

- Engine đọc config/policy_default.json vào đối tượng default_obj. Nếu tồn tại config/policy_overrides.json, đọc vào ov_obj. Sau đó engine lọc ov_obj chỉ giữ lại các khóa override được phép (như danh sách đã liệt kê ở trên)【28†L33-L40】【28†L41-L47】. Các khóa không nằm trong danh sách whitelist sẽ bị loại bỏ để đảm bảo không có thay đổi trái phép cấu hình cố định.
- Engine thực hiện deep merge: các khóa trong ov_obj (sau khi lọc) sẽ ghi đè lên giá trị tương ứng trong default_obj. Kết quả thu được một dict cấu hình hoàn chỉnh cho phiên chạy, và engine ghi nó ra file out/orders/policy_overrides.json (lưu bản config runtime thực tế đã dùng)【28†L41-L47】. Log của engine sẽ thông báo việc sử dụng baseline + overrides cho phiên.
- Trường hợp đặc biệt: nếu vì lý do nào đó không có file policy mặc định (ví dụ trong một phiên bản cũ, dùng trực tiếp policy_overrides làm full config), thì hàm sẽ thử tìm file policy_for_calibration.json để làm baseline, hoặc dùng luôn policy_overrides.json như config đầy đủ. Tuy nhiên, trong phiên bản hiện tại, luôn giả định có đủ hai file và merge có chọn lọc như trên để bảo vệ các tham số cố định【28†L41-L47】.

Sau khi hợp nhất, engine tải file config runtime này và validate nó theo schema định nghĩa (dùng Pydantic model PolicyOverrides trong scripts/engine/schema.py). Việc validate nhằm đảm bảo không thiếu trường bắt buộc hoặc sai kiểu dữ liệu – nếu có sẽ báo lỗi ngay lập tức【28†L43-L47】. Kết quả cuối cùng là một đối tượng cấu hình chiến lược pol_obj (thường gọi là tuning trong code) sẵn sàng cho bước ra quyết định.

Calibrations và guardrails: Bên cạnh phần override linh hoạt bởi AI, hệ thống còn có các tham số được calibrate tự động từ dữ liệu. Ví dụ, sau khi tải dữ liệu giá, engine có thể tính toán các ngưỡng phần trăm cho biến động (quantiles) hay ATR để điều chỉnh tham số ngưỡng tín hiệu (q_add, q_new, ngưỡng volatility guard,...). Những calibrator này chạy ngầm trong quá trình chuẩn bị policy runtime, nhằm đảm bảo tham số phù hợp với điều kiện thị trường hiện tại (thay vì cố định). Tuy nhiên, các calibrations này được thiết kế chỉ tác động lên tham số kỹ thuật (ví dụ ngưỡng dựa trên phân vị thống kê) chứ không thay đổi triết lý chiến lược cốt lõi. Trong khi đó, các thay đổi AI hàng ngày (override) cũng chịu ràng buộc cứng về biên độ và danh sách khóa như trên, và thậm chí có cơ chế ngắt (kill-switch) khi thị trường quá xấu: ví dụ nếu kích hoạt chế độ risk-off cứng, hệ thống sẽ bỏ qua mọi override để trở về policy baseline an toàn【30†L63-L71】. Nhờ các lớp bảo vệ này, việc kết hợp tham số cố định, calibrations tự động và điều chỉnh AI diễn ra một cách có kiểm soát.

Tóm lại, policy của Broker GPT bao gồm một cấu hình nền tảng ổn định và một lớp tùy biến hạn chế. Mọi thay đổi ngắn hạn đều được giới hạn trong khuôn khổ an toàn (whitelist khóa, giới hạn giá trị, yêu cầu giải thích) và có thể bị vô hiệu hóa khi thị trường biến động quá tiêu cực. Điều này giúp engine vừa vững chắc về mặt chiến lược, vừa có khả năng thích ứng nhất định với hoàn cảnh mới.

Nhận Diện Chế Độ Thị Trường (Market Regime Detection)

Trước khi đưa ra bất kỳ lệnh cụ thể nào, engine cần hiểu bức tranh toàn cảnh của thị trường – hay gọi là xác định chế độ thị trường hiện tại. Kết quả của bước này sẽ ảnh hưởng trực tiếp đến cách phân bổ nguồn lực (ngân sách, số lượng lệnh) và khẩu vị rủi ro cho phiên giao dịch.

Hàm chính đảm nhiệm việc này là get_market_regime(session_summary, sector_strength, tuning) trong scripts/order_engine.py. Hàm này tạo ra một đối tượng MarketRegime (được định nghĩa trong scripts/engine/schema.py hoặc nội tuyến bằng dataclass) bao gồm các thuộc tính sau【30†L67-L75】:

- Thông tin phiên giao dịch: Giai đoạn phiên (phase) hiện tại – ví dụ pre-market, morning, lunch, afternoon, ATC, post-market – và cờ in_session cho biết đang trong giờ giao dịch hay không. Thông tin này lấy từ session_summary_df (nếu có) hoặc suy ra từ đồng hồ hệ thống【30†L67-L70】.
- Chỉ báo thị trường tổng thể: % thay đổi của VN-Index trong phiên hiện tại (index_change_pct), chỉ số breadth của thị trường (breadth_hint – ví dụ tỷ lệ cổ phiếu trên MA50), và đôi khi cả trend_strength (sức mạnh xu hướng chính, như so với MA200). Từ các chỉ báo này cùng dữ liệu biến động, hệ thống xác định cờ risk_on (True/False) – thị trường có đang ở trạng thái thuận lợi (risk-on) để mua vào hay đang rủi ro cao (risk-off) nên hạn chế mua. Đi kèm là một điểm số thị trường tổng hợp (market_score trong khoảng 0-1) thể hiện xác suất hay mức độ tích cực của thị trường (xác suất risk-on) dựa trên mô hình nội bộ. Ví dụ, nếu VN-Index tăng mạnh, breadth cao -> market_score cao, ngược lại nếu Index giảm sâu, breadth kém -> market_score thấp【30†L69-L72】.
- Ngân sách và hạn mức lệnh: Các tham số quan trọng từ policy (sau khi đã merge override và calibrations) như buy_budget_frac, add_max, new_max được đưa vào MarketRegime【30†L69-L72】. Đây là những giới hạn mà bước quyết định lệnh phải tuân theo: tổng ngân sách dành cho mua, tối đa bao nhiêu lệnh mua bổ sung và mua mới. Nếu override AI có thay đổi các giá trị này trong ngày thì giá trị hiệu lực sau cùng cũng nằm ở đây.
- Trọng số mô hình và ngưỡng điểm: weights (trọng số các thành phần tính điểm tín hiệu) và thresholds (các ngưỡng điểm như base_add, base_new, trim_th đã qua hiệu chỉnh). Những giá trị này cũng đến từ policy (sau khi áp dụng bất kỳ calibrator nào). Ngoài ra, các bias sector_bias, ticker_bias (nếu có) từ override AI sẽ được đính kèm vào đây để dùng trong tính điểm【30†L69-L72】.
- Thông số vi mô về giá & khớp lệnh: Bao gồm các cấu hình về pricing (cách chọn giá đặt lệnh mua/bán tùy chế độ thị trường, ví dụ risk-on có thể đặt giá cao hơn để mau khớp, risk-off đặt giá thận trọng hơn) và sizing (quy tắc phân bổ vốn, ví dụ tỷ trọng cho mỗi lệnh mới/add, có tái sử dụng tiền bán hay không) và execution (tham số vi mô về khớp lệnh như độ trượt giá cho phép, cách điều chỉnh giá nếu biến động mạnh, v.v.). Những phần này lấy từ policy mặc định và có thể được tinh chỉnh nhẹ bởi dữ liệu (ví dụ trượt giá ATR), nhưng nói chung cố định trong mỗi lần chạy【30†L71-L73】.
- Sức mạnh ngành: Kết quả phân tích ngành sector_strength_df được tổng hợp vào sector_strength_rank – một dict điểm 0-1 cho từng ngành thể hiện ngành nào đang mạnh/yếu tương đối. Engine có thể dùng thông tin này để ưu tiên ngành mạnh (ví dụ mua ưu tiên ngành đang dẫn dắt thị trường).
- Tín hiệu vĩ mô (nếu có): Các thông số như epu_us_percentile (độ cao của chỉ số bất ổn chính sách kinh tế Mỹ so với lịch sử), spx_drawdown_pct (mức sụt giảm hiện tại của S&P 500), dxy_percentile (sức mạnh USD) cũng được lưu trong MarketRegime nếu bước pipeline vĩ mô có chạy. Những thông tin này giúp engine quyết định có kích hoạt chế độ phòng thủ không. Ví dụ, nếu EPU ở phân vị rất cao (rủi ro vĩ mô lớn), policy có thể cấu hình để guardrail hạn chế lệnh mua mới.
- Chế độ Neutral (Trung tính thích ứng): Một đặc tính nâng cao của Broker GPT là chế độ Neutral-Adaptive. Nếu thị trường không rõ xu hướng (không đủ điều kiện gọi risk-on, nhưng cũng chưa đến mức risk-off cứng), engine có thể đặt trạng thái is_neutral = True. MarketRegime chứa các trường liên quan: neutral_state (một số config nội bộ cho chế độ neutral), và các danh sách trống cho đến khi được sử dụng: neutral_partial_tickers, neutral_override_tickers, neutral_accum_tickers – sẽ được điền trong quá trình quyết định lệnh nếu áp dụng kịch bản neutral. Mục đích của neutral mode là thận trọng khi thị trường “lưng chừng”: vẫn xem xét cơ hội nhưng giảm quy mô. Ví dụ, trong neutral mode:
  - Engine cho phép mua mới nhưng với quy mô nhỏ (partial entry) – các lệnh mua mới có thể chỉ mua một phần nhỏ (được đánh dấu là PARTIAL_ENTRY).
  - Hạn chế tối đa việc mua bổ sung (add_max có thể giảm xuống rất thấp) vì thị trường chưa đủ mạnh để tăng thêm vị thế lớn.
  - Nếu có mã nào rất tiềm năng vượt qua ngưỡng thông thường (override), engine có thể vẫn mua (đánh dấu neutral_override) nhưng với sự thận trọng.
  - Sau khi quyết định lệnh, những mã nào thuộc các trường hợp đặc biệt neutral sẽ được ghi vào các danh sách neutral_partial_set, neutral_override_set, neutral_accum_set để báo cáo và tránh xử lý lặp ở phiên sau nếu chưa thay đổi trạng thái【30†L77-L85】.

Đối tượng MarketRegime gói gọn toàn bộ bối cảnh thị trường và tham số chiến lược tại thời điểm hiện tại. Sau bước này, engine “biết” thị trường đang ở trạng thái nào (phân loại thô: risk-on, neutral, hay risk-off) và điều đó quyết định cách thức ra lệnh: nếu risk-on thì sẵn sàng giải ngân mạnh hơn, risk-off thì phòng thủ (giảm mua hoặc ngừng mua), neutral thì ở giữa (giải ngân thăm dò). 

Ví dụ, khi risk_on = True: buy_budget_frac có thể ~10-15% NAV, add_max/new_max ở mức cao, chiến lược đặt giá mua có thể chủ động hơn (đặt sát giá thị trường để dễ khớp) và giá bán tham vọng hơn (kỳ vọng thị trường thuận lợi)【30†L87-L90】. Ngược lại nếu risk_on = False (kích hoạt risk-off): ngân sách mua có thể giảm xuống rất thấp (ví dụ 2-3% NAV), thậm chí hệ thống có thể kill-switch ngừng hẳn lệnh mua mới; cách đặt giá thì thận trọng (mua giá thấp, bán giảm giá để thoát nhanh)【30†L87-L90】. Trong trường hợp neutral, các thông số ở mức trung bình cộng với cơ chế partial như đã mô tả.

Engine dựa trên các chỉ báo định lượng và ngưỡng trong policy để xác định những trạng thái này. Chẳng hạn, nếu VN-Index giảm quá ngưỡng risk_off_index_drop_pct đồng thời breadth thị trường < risk_off_breadth_floor và các chỉ báo khác đều xấu, engine sẽ cho rằng thị trường rất xấu -> đặt risk_on = False và thậm chí nếu vượt ngưỡng “severe” thì scale ngân sách = 0 (không mua gì)【30†L91-L94】. Ngược lại, nếu đa số tín hiệu tích cực vượt ngưỡng risk_on_threshold, engine đặt risk_on = True. Trạng thái neutral xảy ra khi các chỉ số chỉ vừa đủ không vi phạm mức risk-off cứng nhưng cũng chưa đạt điều kiện risk-on rõ ràng – khi đó engine scale ngân sách ở mức giữa (ví dụ 50% của full risk-on) tương ứng độ mạnh yếu của tín hiệu【30†L91-L94】.

Kết quả của bước nhận diện chế độ sẽ được log ra phần diagnostics cuối quá trình chạy. File orders_analysis.txt thường có các dòng như “Regime risk_on: True/False”, “Buy budget frac: X (effective Y)”, “Top sectors: ...”, “Risk-on probability: Z” v.v., tất cả đều dựa trên đối tượng MarketRegime vừa tính được【30†L91-L94】. Những thông tin này giúp người dùng hiểu bối cảnh thị trường mà engine nhận định trước khi xem chi tiết từng lệnh.

Thuật Toán Ra Quyết Định Lệnh Mua/Bán

Đây là phần lõi quyết định của Broker GPT Engine. Sau khi pipeline cung cấp dữ liệu và MarketRegime xác định bối cảnh, engine sẽ duyệt qua từng mã cổ phiếu để tính điểm tín hiệu và phân loại hành động (mua mới, mua thêm, giữ, giảm bán, hoặc bán hết). Quá trình này có thể chia thành các bước chính:

A) Tính Điểm Tín Hiệu (Conviction Score) cho từng mã

Engine xử lý danh mục hiện tại trước tiên (các mã người dùng đang nắm giữ), sau đó mới xem xét đến các mã chưa có. Lý do là ưu tiên quyết định cho những mã đã sở hữu (xem có cần mua thêm hay bán bớt) trước khi quyết định mở vị thế mới. 

Cho mỗi mã đang có trong danh mục:

1. Lấy dữ liệu đặc trưng: Engine lấy dòng tương ứng của mã đó từ snapshot_df (giá hiện tại, số lượng đang giữ, v.v.) và metrics_df (các chỉ báo đã tính: RSI, ATR%, Beta, thanh khoản, v.v.). Nếu thiếu metric quan trọng nào, engine sẽ ghi nhận cảnh báo (ví dụ thiếu RSI do thiếu dữ liệu lịch sử). Đồng thời, engine bổ sung các đặc trưng kỹ thuật ngắn hạn từ presets_df – chẳng hạn giá trị MA20, MA50 cho mã – ghép vào dữ liệu snapshot. Điều này nhằm có đủ thông tin để tính các tín hiệu cắt MA, quá mua/quá bán... (vì snapshot ban đầu chỉ có giá hiện tại, cần thêm MA để biết giá đang trên hay dưới MA)【33†L101-L104】.

2. Tính vector feature: Engine gọi hàm compute_features(ticker, snapshot, metrics, normalizers) để tạo ra một feature vector mô tả trạng thái mã cổ phiếu đó. Các feature có thể bao gồm:
   - % thay đổi giá so với phiên trước,
   - Khoảng cách giá hiện tại so với MA20, MA50 (% trên hay dưới đường MA),
   - RSI14 hiện tại và so với ngưỡng trung tính (50),
   - Biến động ATR14% (tương quan biên độ dao động so với giá),
   - Xếp hạng động lượng (ví dụ MomRetNorm – tỷ suất sinh lợi xếp hạng so với các mã khác),
   - Xếp hạng thanh khoản (LiqNorm – thanh khoản so với thị trường),
   - Mức độ quá mua/bán (như RSI cao/ thấp so với ngưỡng overbought/oversold),
   - … cùng nhiều yếu tố khác tùy thiết kế mô hình.
   Kết quả là một dict feats chứa các feature giá trị số cho mã. Ngoài ra, engine cũng tính thêm một chỉ số quan trọng cho mã đang có: pnl_pct – % lãi/lỗ hiện tại của vị thế (dựa trên giá hiện tại so với giá vốn AvgCost). Thông tin này cần để kiểm tra các ngưỡng chốt lời/cắt lỗ đã đạt chưa【33†L101-L104】.

3. Tính điểm conviction score: Dựa trên feature vector vừa có, engine tính điểm tín hiệu tổng hợp cho mã thông qua hàm conviction_score(feats, sector, regime, ticker). Điểm này thường được chuẩn hóa trong khoảng -1 đến +1, giá trị càng cao nghĩa là tín hiệu mua càng mạnh, giá trị âm nghĩa là nên bán/giảm. 
   Công thức tính score là kết hợp tuyến tính có trọng số các thành phần: sử dụng regime.weights (bộ trọng số mô hình từ MarketRegime, xuất phát từ policy). Ví dụ, score có thể = w_trend * TrendScore + w_momo * MomentumScore + w_liq * LiqScore + ... cộng tất cả các thành phần (có thể áp dụng điều chỉnh bởi sector_bias/ticker_bias nếu có – ví dụ nếu sector_bias ngành = +0.1 thì cộng thêm vào điểm của mã thuộc ngành đó một lượng tương ứng)【33†L102-L104】. Kết quả cuối cùng, engine nhận được một conviction score cho mã: score[ticker] = sc. Điểm số này phản ánh mức độ hệ thống “tin tưởng” mã đó nên được giữ/mua thêm hay nên bán bớt. 
   Engine lưu lại điểm số vào cấu trúc dữ liệu (dict scores). Đồng thời, toàn bộ feature vector feats cũng được lưu vào feats_all[ticker] để sau này ghi ra file diễn giải (chẳng hạn file orders_reasoning.csv sẽ liệt kê các thành phần điểm cho từng mã)【33†L102-L105】. Nếu trong quá trình tính phát hiện thiếu dữ liệu cho feature nào quan trọng, engine cũng đánh dấu vào regime.diag_warnings để cảnh báo.

Sau khi có score cho mã, engine tiến hành phân loại hành động sơ bộ cho mã đang nắm giữ dựa trên điểm số và ngưỡng:

4. Phân loại hành động ban đầu (classify_action): Engine gọi hàm classify_action(is_holding=True, score=sc, feats, regime, ...) để lấy gợi ý hành động cơ bản cho mã. Logic phân loại như sau (trường hợp is_holding=True):

   - Kiểm tra cắt lỗ cứng: Nếu % P/L hiện tại của mã ≤ -sl_pct_eff (lỗ đã vượt ngưỡng cắt lỗ cho phép, đã hiệu chỉnh – có thể từ policy sl_pct hoặc override), thì trả về hành động "exit" – nghĩa là nên bán toàn bộ ngay để dừng lỗ【33†L109-L113】. Đây là quy tắc ưu tiên cao nhất, ngừng lỗ khẩn cấp.
   - Kiểm tra chốt lời cứng: Nếu % P/L hiện tại ≥ tp_pct_eff (đạt ngưỡng mục tiêu chốt lời), trả về "take_profit" – tức chốt toàn bộ vị thế để khóa lợi nhuận【33†L109-L113】.
   - Kiểm tra tín hiệu cắt MA xấu: Nếu cấu hình cho phép (ví dụ exit_on_ma_break = true trong policy), engine kiểm tra xem giá đã cắt xuống dưới MA50 và RSI hiện tại < ngưỡng exit_ma_break_rsi hay chưa. Nếu có, đây là tín hiệu kỹ thuật xấu báo hiệu đảo chiều giảm:
     * Mặc định trường hợp này trả về "exit" (bán hết)【33†L111-L117】.
     * Tuy nhiên, có một số điều kiện giảm nhẹ: nếu conviction score của mã vẫn đủ cao (≥ ngưỡng exit_ma_break_score_gate) và tỷ lệ Reward/Risk còn tốt (≥ 1.0 chẳng hạn), thì engine hạ hành động từ exit xuống "trim" (bán bớt một phần thay vì bán hết)【33†L111-L117】. 
     * Tương tự, nếu mã đó có ticker_bias dương (thiên lệch tích cực) vượt một ngưỡng cho phép, cũng có thể đổi exit thành trim – tức tin tưởng mã này đặc biệt, không bán hết ngay【33†L112-L115】.
     * Hoặc nếu tín hiệu xấu xuất hiện quá sớm (ví dụ ngay đầu phiên sáng), engine cũng có thể tránh panic sell: hạ từ exit xuống trim để chờ thêm tín hiệu xác nhận【33†L113-L116】.
     * Ngược lại, nếu biến động đang quá cao (ATR cao bất thường), thì giữ quyết định exit thẳng (vì biến động cao đồng nghĩa rủi ro thêm nếu cố nắm giữ)【33†L114-L117】.
     * Sau khi xét các điều kiện, nếu không có yếu tố giảm nhẹ nào, mặc định giá cắt MA50 + RSI thấp => action = "exit".
   - Nếu không rơi vào trường hợp bán ngay ở trên:
     * Nếu score >= base_add (điểm tín hiệu đủ mạnh để tăng thêm vị thế) thì trả về "add" – tín hiệu mua bổ sung cổ phiếu này【33†L117-L120】.
     * Nếu score <= trim_th (điểm yếu đáng kể, ngưỡng có thể âm) hoặc có các dấu hiệu suy yếu khác (ví dụ giá cắt xuống MA20 và RSI thấp, hoặc MACD âm kèm RSI thấp), thì trả về "trim" – đề xuất bán giảm bớt một phần vị thế để giảm rủi ro【33†L118-L121】.
     * Nếu không thỏa mãn điều kiện nào đặc biệt, trả về "hold" – tiếp tục giữ, không hành động với mã này【33†L118-L121】.

   Đối với mã chưa có trong danh mục (sẽ chạy vòng sau), logic classify_action(is_holding=False) đơn giản hơn:
   - Nếu score >= base_new (điểm rất cao vượt ngưỡng mua mới) thì trả về "new" – tín hiệu đủ mạnh để mở vị thế mua mã này【33†L122-L125】.
   - Ngược lại, trả về "ignore" – bỏ qua mã (không mua).

   Các ngưỡng base_add, base_new, trim_th... ở trên đều xuất phát từ policy (có thể đã calibrate theo chi phí giao dịch). Thông thường, base_add ~ 0.6-0.7, base_new ~ 0.8-0.9, còn trim_th có thể âm (ví dụ -0.2) để chủ động cắt giảm vị thế khi điểm số chuyển âm【33†L125-L128】.

5. Áp dụng logic dừng lỗ/chốt lời từng phần (stateless stops): Sau khi có default_action từ classify_action cho mã đang có, engine tiếp tục kiểm tra xem có cần ghi đè hành động này do các quy tắc quản lý vị thế nhiều bước hay không. Cụ thể, policy có thể định nghĩa các mức chốt lời nhiều phần hoặc dừng lỗ từng phần cho mỗi mã, thay vì tất cả hoặc không. Ví dụ:
   - Nếu mã đang lỗ nặng đạt đến một tỷ lệ nhất định của ngưỡng cắt lỗ (ví dụ 80% mức lỗ tối đa cho phép, cấu hình qua sl_step2_trigger), engine sẽ bán hết ngay (exit) mặc dù default_action có thể chỉ là trim hoặc hold, vì coi như đã chạm ngưỡng stop-loss khẩn cấp. Trường hợp này engine gán action = 'exit' kèm note SL_STEP2, và đánh dấu trạng thái rằng mã này đã bán do step2 để lần sau không lặp lại【33†L128-L131】.
   - Nếu lỗ đạt mức trung bình (ví dụ 50% ngưỡng cắt lỗ, sl_step1_trigger) và trước đó chưa bán phần nào, engine có thể bán một phần (trim) tỷ lệ nhất định (ví dụ 25% vị thế, cấu hình sl_step1_frac), gán note SL_STEP1, đồng thời đánh dấu trạng thái đã thực hiện step1 cho mã đó【33†L129-L134】.
   - Tương tự với lãi: nếu mã lãi vượt ngưỡng tp_pct_eff và chưa từng chốt lời phần nào, và có cấu hình chốt lời một phần (ví dụ tp1_frac = 0.5 tức chốt 50%), engine sẽ thực hiện chốt lời một phần: đặt action = 'take_profit' nhưng với tp_frac 50% (nghĩa là bán một nửa), kèm note TP1 (chẳng hạn TP1_ATR nếu dựa ATR). Sau đó đánh dấu đã thực hiện tp1 để không lặp lại【33†L131-L134】.
   - Ngoài ra, nếu phát hiện tín hiệu động lượng rất yếu (ví dụ RSI rất thấp, giá dưới MA20/MA50) mà default_action không phải exit/TP, engine có thể quyết định trim một phần nhỏ (ví dụ 30%) với note MOM_WEAK – giảm vị thế để phòng rủi ro do động lượng suy yếu【34†L134-L138】.

   Những kiểm tra trên tạo ra một quyết định meta (meta_decision) có thể khác với default_action. Nếu có meta_decision:
   - Engine sẽ ghi đè hành động cuối cùng cho mã đó: act[ticker] = meta_decision.action【34†L136-L139】.
   - Lưu chi tiết meta vào sell_meta[ticker] (nếu là lệnh bán một phần hoặc bán hết do stop-loss/take-profit đặc biệt). Nếu đó là lệnh dừng lỗ (stop order) có khái niệm thời hạn (TTL), engine đặt TTL override cho mã này – nghĩa là lệnh bán này có hiệu lực trong một khoảng thời gian xác định (ví dụ TTL vài phút; điều này để nếu lệnh không khớp ngay có thể tự hủy)【34†L136-L139】.
   - Cập nhật position_state cho mã: ví dụ đặt cờ sl_step_hit_50=True hay tp1_done=True để phiên sau engine biết mã này đã thực hiện step1, từ đó sẽ thực hiện step2 hay không. Trạng thái này sẽ được ghi ra file out/orders/position_state.csv khi kết thúc để lưu vết【34†L136-L140】.

   Nếu không có meta_decision nào áp dụng, engine giữ nguyên default_action. Kết thúc vòng lặp qua các mã đang giữ, ta có danh sách act[] cho mỗi mã trong danh mục, giá trị có thể là: hold, add, trim, exit (bán hết), hoặc take_profit (bán phần lớn, tương tự exit nhưng do chốt lời).

B) Xử lý các mã chưa có trong danh mục (nhưng thuộc vũ trụ phân tích):

- Engine thực hiện tương tự: lấy snapshot + metrics cho mã, tính toán feats và tính score y hệt như với mã đang có【34†L143-L147】. Điểm số này cũng chịu ảnh hưởng của ticker_bias/sector_bias nếu có (ví dụ toàn ngành được +0.1 thì mã này cũng +0.1 vào score).
- Lấy ngưỡng riêng nếu có override cho mã này (policy cho phép định nghĩa ticker_overrides – ví dụ một số mã có thể có ngưỡng base_new riêng).
- Gọi classify_action(False, score, ...). Nếu kết quả là "new", engine đánh dấu act[ticker] = "new" – mã này là ứng viên mua mới【34†L143-L147】. Nếu kết quả "ignore" thì bỏ qua mã (không thêm vào act).
- Engine lưu scores[ticker] = sc cho các mã này, và feats_all[ticker] = feats để sau ghi file reasoning. Ngoài ra, nếu trong quá trình tính toán cần chuẩn bị thông tin cắt lỗ cho mã mới (ví dụ tính mức dừng lỗ đề xuất dựa trên ATR – gọi là tp_sl_info), engine sẽ lưu vào tp_sl_map cho mã đó để dùng ở bước tính khối lượng lệnh sau【34†L143-L148】.

Đến đây, engine đã có một danh sách hành động sơ bộ cho toàn bộ mã trong vũ trụ:
- Đối với mã đang có: mỗi mã được gắn nhãn hành động hold, add, trim, exit hoặc take_profit (take_profit thường là bán phần, exit là bán hết).
- Đối với mã chưa có: hoặc là new (ứng viên mua mới) hoặc không có hành động (bỏ qua).

Bộ Lọc Hành Động & Kiểm Soát Rủi Ro

Trước khi thực sự chuyển sang bước tính toán quy mô lệnh và giá, engine áp dụng một loạt bộ lọc loại trừ trên danh sách hành động vừa xác định nhằm đảm bảo tuân thủ các điều kiện thị trường và quy tắc kiểm soát rủi ro:

- Bộ lọc giá trần: Bất kỳ hành động mua (add hoặc new) nào nếu giá hiện tại của mã đã quá gần giá trần biên độ trong phiên sẽ bị loại bỏ. Cụ thể, nếu Price >= near_ceiling_pct * BandCeiling (ví dụ giá cổ phiếu đạt 98% giá trần ngày) thì engine đổi hành động từ mua thành hold, bỏ lệnh mua đó【34†L153-L156】. Lý do: tránh mua đuổi các mã đã tăng kịch trần (khả năng khớp thấp, rủi ro cao). Những mã bị lọc bởi tiêu chí này được thêm vào danh sách debug filters["near_ceiling"] kèm giải thích (ví dụ “(ADD) price 49.0 within 0.98 of ceiling 50.0”).

- Bộ lọc thị trường (market guard): Nếu thị trường chung đang kích hoạt chế độ bảo vệ (guard) khiến việc mua mới không an toàn, engine sẽ hoãn các lệnh mua mới. Điều này phản ánh qua cờ như guard_new trong MarketRegime hoặc trực tiếp qua scale ngân sách = 0. Chẳng hạn, policy có thể đặt guard_new=True khi market_score thấp dưới ngưỡng mềm: khi đó dù một số mã có điểm cao, engine vẫn chặn lệnh "new" để chờ thị trường cải thiện【34†L153-L156】. Cụ thể, nếu risk_on=False hoặc neutral với dấu hiệu xấu, new_max có thể đã bị set = 0. Engine sẽ lọc bỏ hoặc không tạo lệnh new nào. Những mã bị loại do điều kiện thị trường chung xấu sẽ chuyển thành hold và debug filter ghi note “market filter active – defer adding until trend/breadth improves”. (Trường hợp thị trường rất xấu – chế độ severe risk-off – engine có thể đã đặt scale=0 cho ngân sách, tức hoàn toàn không mua mới, đây chính là kill-switch đã nói).

- Bộ lọc thanh khoản yếu: Nếu một mã có thanh khoản quá kém so với chuẩn trong policy, engine sẽ loại bỏ lệnh mua vào mã đó. Cụ thể, policy có tham số min_liq_norm – yêu cầu xếp hạng thanh khoản tối thiểu. Nếu mã có LiqNorm < min_liq_norm, engine coi thanh khoản không đủ đảm bảo cho giao dịch an toàn, và sẽ không mua mã đó. Mã vi phạm bị loại và ghi vào debug filter filters["liquidity"]. (Trong code, check này có thể gián tiếp: ví dụ nếu min_liq_norm > 0, engine yêu cầu cột LiqNorm phải tồn tại và >0 để mua; nếu thiếu coi như không đạt, loại bỏ).

- Giới hạn số lượng lệnh: Cuối cùng, engine kiểm soát rằng tổng số lệnh add và new không vượt quá add_max và new_max tương ứng:
  * Tập hợp tất cả mã có hành động "add" thành danh sách add_names. Nếu kích thước danh sách > add_max, engine sẽ ưu tiên giữ lại các mã điểm cao hơn. Thực hiện bằng cách sắp xếp danh sách theo score giảm dần rồi cắt chỉ lấy top add_max mã. Các mã bị vượt ngưỡng sẽ bị chuyển thành hold (bỏ lệnh add). 
  * Tương tự với hành động "new": lấy danh sách new_names, sắp xếp theo điểm từ cao xuống và chỉ giữ lại top new_max mã. Những mã dù có điểm vượt ngưỡng nhưng nằm ngoài top sẽ không được mua phiên này. Engine có thể đưa các mã “new” bị cắt này vào dạng watchlist để theo dõi sau. (Trong code, ví dụ: new_sorted = sorted(new_names, key=lambda x: score, reverse=True)[:regime.new_max] rồi chỉ những mã trong new_sorted mới thành lệnh【34†L156-L160】). 

Những bộ lọc trên đảm bảo kỷ luật giao dịch: không mua đuổi giá trần, không vi phạm nguyên tắc thị trường xấu không mua, tránh mã thanh khoản kém, và giới hạn số lệnh để tập trung vốn. Mọi mã bị loại bởi filter thường được ghi vào file out/orders_filtered.csv hoặc orders_watchlist.csv kèm lý do để người dùng biết【34†L153-L160】. Đặc biệt, các mã “new” bị loại có thể xuất hiện trong orders_watchlist.csv – đây là danh sách các mã đáng chú ý (điểm cao) nhưng không được mua do vi phạm một số điều kiện vi mô hoặc vượt hạn mức, người dùng có thể theo dõi thủ công.

Sau bước lọc, danh sách hành động cuối cùng (final actions) đã sẵn sàng cho bước tính toán chi tiết lệnh.

Phân Bổ Ngân Sách và Tính Khối Lượng Lệnh (Sizing & Execution)

Ở giai đoạn này, engine biết những mã nào sẽ mua (add hoặc new) và bán (trim, exit, take_profit). Tiếp theo cần quyết định mua bao nhiêu, bán bao nhiêu và giá nào cho từng lệnh, tuân thủ ngân sách và các nguyên tắc khớp lệnh.

Tính ngân sách mua khả dụng:
- NAV: Engine ước tính tổng giá trị tài sản danh mục hiện tại (Net Asset Value) dựa trên giá thị trường hiện tại của các cổ phiếu đang nắm giữ. NAV này có trong out/portfolio_pnl_summary.csv (cột TotalMarket). Giả sử NAV = X (đơn vị nghìn đồng).
- Ngân sách mua thô = buy_budget_frac * NAV. Đây là số tiền tối đa dự kiến chi cho các lệnh mua (gồm cả mua mới và mua bổ sung) trong phiên, sau khi đã tính đến chế độ risk-on/off (ví dụ risk-off => buy_budget_frac rất nhỏ)【34†L167-L170】.
- Tái sử dụng tiền bán (nếu cho phép): Policy có tham số reuse_sell_proceeds_frac (tỷ lệ % tiền bán ra có thể tái dùng để mua trong cùng phiên). Engine tính tổng giá trị dự kiến thu được từ các lệnh bán (sell_candidates). Nếu policy cho phép tái sử dụng, engine cộng thêm một phần tiền này vào ngân sách mua. Ví dụ: nếu dự kiến bán thu về 500 (nghìn đồng) và reuse_sell_proceeds_frac = 0.5 (50%), thì cộng thêm 250 vào ngân sách mua. Kết quả thu được ngân sách mua cuối cùng gọi là target_gross_buy【34†L168-L171】.

Phân bổ ngân sách cho các lệnh mua:
Engine xác định phương pháp phân bổ dựa trên tham số allocation_model trong policy (có thể là 'mean_variance', 'risk_budget', hoặc mặc định 'proportional')【35†L1-L9】:

- Nếu allocation_model = 'mean_variance': Engine áp dụng tối ưu hóa danh mục theo lý thuyết Trung bình-Phương sai (Markowitz). Cụ thể, lấy dữ liệu giá lịch sử của các mã sẽ mua (cộng thêm VN-Index làm đại diện thị trường) từ prices_history_df, tính log-returns và ma trận hiệp phương sai (covariance). Dùng kỹ thuật shrinkage Ledoit–Wolf để ước lượng ma trận cov ổn định (có tham số điều chỉnh cov_reg). Sau đó ước lượng suất sinh kỳ vọng (expected returns) cho từng mã, có thể dùng mô hình Black–Litterman nếu tích hợp. Thực tế, hàm compute_expected_returns() trong portfolio_risk.py sẽ tính toán phần này, kết hợp điểm số hiện tại (score) thành “quan điểm” (view) về kỳ vọng lợi nhuận, cùng tham số thị trường chung (risk-free rate, market premium)【35†L1-L5】. Cuối cùng, engine giải bài toán tối ưu (qua hàm solve_mean_variance_weights) để tìm trọng số vốn tối ưu cho mỗi mã mua, tối ưu hóa tỷ lệ Sharpe hoặc theo rủi ro mục tiêu. Những trọng số này sau đó được nhân với ngân sách để ra số tiền cho mỗi mã.

- Nếu allocation_model = 'risk_budget': Engine dùng phương pháp Risk Budgeting đơn giản. Mỗi mã được phân tiền tỷ lệ thuận với điểm số / rủi ro của nó. Cụ thể, code _allocate_risk_budget tính cho từng mã một trọng số = score / (γ * σ^2), trong đó σ là độ biến động (có thể dùng độ lệch chuẩn hoặc ATR) của mã, còn γ là tham số độ e ngại rủi ro (risk_aversion)【35†L6-L10】. Sau đó chuẩn hóa trọng số để tổng thành 100% ngân sách. Kết quả: mã nào điểm cao và biến động thấp sẽ được phân nhiều vốn hơn, đảm bảo đóng góp rủi ro giữa các mã gần cân bằng (ý tưởng mỗi mã một phần “risk budget”).

- Nếu allocation_model khác hoặc không được chỉ định (mặc định): Engine dùng cách phân bổ tỉ lệ theo điểm số (proportional). Tức là chia ngân sách theo tỷ lệ điểm tín hiệu của mỗi mã. Mã có điểm cao hơn nhận nhiều vốn hơn. Tuy nhiên, để tránh dồn quá nhiều vào một mã điểm cao nhất, hệ thống có thể áp dụng một hàm phân phối mềm như softmax với nhiệt độ τ (tham số sizing.softmax_tau) để làm mượt phân bố. Softmax sẽ làm giảm chênh lệch giữa điểm cao nhất và các điểm khác tùy theo τ. Tham số τ này có thể được calibrate trước để đạt độ phân tán mục tiêu (ví dụ độ phân bổ không quá tập trung). Thực tế, hàm allocate_proportional sẽ thực hiện việc chuyển điểm thành trọng số (có thể thông qua softmax)【35†L7-L10】.

Sau khi quyết định tỷ trọng vốn (trọng số) cho từng mã mua, engine tính giá trị tiền phân bổ cho mỗi lệnh:
- Đối với lệnh mua mới (new): số tiền = trọng số * target_gross_buy.
- Đối với lệnh mua bổ sung (add): cũng tương tự, nhưng có thể có điều chỉnh nếu policy quy định khác giữa vốn cho new vs add. Thông thường engine gộp chung các mã mua (new + add) để phân bổ tổng thể. Tuy nhiên, policy có tham số đảm bảo cân bằng: ví dụ không để vốn dồn hết vào mã new mà bỏ qua add (vì add là những mã có sẵn vị thế, thường ít rủi ro hơn new). Chi tiết này tùy thuộc implement – có thể engine sẽ ưu tiên add trước new hoặc ngược lại tùy chiến lược. (Trong thực tế, việc sắp xếp top new_max đã giới hạn số mã new, nên add và new đều trong danh sách cuối cùng).

Tính khối lượng (Quantity) và giá đặt lệnh:
- Khi đã có số tiền dự kiến cho mỗi lệnh mua, engine chia cho giá cổ phiếu để ra số lượng cổ phiếu định mua. Sau đó, áp dụng quy tắc làm tròn phù hợp:
  * Làm tròn xuống số lượng sao cho thỏa bước lot 100 cổ phiếu (HOSE quy định khối lượng giao dịch phải bội số của 100). Engine có hàm hỗ trợ để làm tròn khối lượng.
  * Đảm bảo số lượng không vượt các giới hạn trong policy: ví dụ không mua quá max_position_percent của NAV cho một mã (giới hạn tỷ trọng tối đa mỗi mã), cũng như không vượt quá tỷ lệ thanh khoản (ví dụ không mua hơn X% khối lượng giao dịch trung bình ngày của mã để tránh ảnh hưởng giá).
  * Nếu sau làm tròn mà số tiền dùng không hết (có leftover vốn dư do làm tròn), engine có thể thực hiện phân bổ lại số dư đó: ví dụ lặp lại vòng phân bổ thêm 1 lot cho các mã còn room theo thứ tự điểm cao cho đến khi hết tiền dư hoặc không thể phân bổ (engine có thể lặp tối đa ~32 vòng để dùng hết leftover)【36†L1-L4】. Điều này đảm bảo tối ưu sử dụng vốn, tránh để sót vốn chưa dùng nếu vẫn có thể mua thêm vài lô cổ phiếu.
- Với lệnh bán (trim/exit): số lượng bán thường đã được xác định từ bước quyết định (ví dụ trim 25% hay exit toàn bộ). Engine chỉ cần chuyển tỷ lệ đó thành số cổ phiếu để bán. Cũng phải làm tròn cho khớp bội số 100 (nếu bán hết mà số lượng đang nắm không tròn 100, có thể bán hết lẻ).
- Giá đặt lệnh: Engine xác định mức giá hợp lý để đặt cho mỗi lệnh:
  * Nếu là lệnh mua: Có chiến lược đặt giá tùy thuộc chế độ market regime. Ví dụ: khi risk-on (thị trường tích cực), engine có thể đặt giá tương đối cao (gần giá hiện tại hoặc thậm chí giá trần - 1 tick) để tăng khả năng khớp mua nhanh. Khi risk-off, đặt giá thận trọng thấp (gần giá sàn hơn) để chỉ mua nếu giá thật sự tốt. Thông số định ra cách đặt giá nằm trong policy.pricing. Thường bao gồm: tham số như buy_up_ticks hoặc phần trăm ATR để cộng/trừ giá.
  * Nếu là lệnh bán: Tương tự, risk-on có thể đặt giá bán tham vọng cao hơn (vì kỳ vọng giá còn lên), risk-off đặt giá thấp để thoát nhanh. Ngoài ra, nếu lệnh bán do stop-loss khẩn cấp, giá có thể đặt sát giá sàn để chắc chắn khớp.
  * Engine cũng phải làm tròn giá theo bước giá (tick size) của HOSE. Ví dụ, giá cổ phiếu 18.x thì bước nhảy 50 đồng, trên 50 thì 100 đồng... Hàm round_to_tick() (có trong scripts/utils.py) được dùng để làm tròn giá cho hợp lệ【25†L21-L25】. 
  * Một số tham số vi mô khác: tránh đặt lệnh bán ở giá dưới tick minimal slip nếu không cần, v.v., đều tuân theo config execution.

Gom danh sách lệnh và kết xuất:
- Engine hợp nhất các lệnh bán và mua vào danh sách orders. Thứ tự ưu tiên thường là lệnh MUA trước rồi đến BÁN (hoặc ngược lại tùy yêu cầu đầu ra, nhưng README cho biết file orders_final sẽ liệt kê BUY trước rồi SELL)【3†L35-L40】.
- Mỗi mục lệnh bao gồm: Ticker, Loại (Side: BUY/SELL), Số lượng, Giá giới hạn (LimitPrice). 

Engine cũng thu thập thêm thông tin để ghi vào các file phụ:
- orders_reasoning.csv: chứa chi tiết điểm số và các thành phần tính điểm cho mỗi mã (lấy từ feats_all và scores). File này giúp người dùng hiểu lý do vì sao một mã được mua hay bán (ví dụ các cột: TrendScore, MomScore, LiqScore, tổng điểm, bias, v.v.).
- orders_quality.csv: chứa các đánh giá về chất lượng lệnh và vi mô khớp lệnh. Ví dụ: 
  * Xác suất khớp lệnh (FillProb) và tỷ lệ khớp kỳ vọng (FillRateExp) dựa trên thanh khoản: Engine ước tính dựa trên khối lượng đặt so với thanh khoản trung bình (ADTV) xem khả năng khớp là bao nhiêu. Lệnh quá lớn so với thị trường sẽ có FillProb thấp.
  * Độ trượt giá dự kiến: Engine sử dụng mô hình trượt giá tuyến tính dựa trên tỷ lệ khối lượng giao dịch (có tham số pricing.tc_roundtrip_frac trong policy) để tính SlipBps (basis points) và SlipPct cho mỗi lệnh. Các giá trị này sau đó dùng tính Expected Return (ExpR) sau phí cho lệnh (đặc biệt để đánh giá lệnh mua có đáng đổi rủi ro phí hay không).
  * Cờ LimitLock: Engine đánh dấu nếu mã đó đang ở phiên giá trần/sàn khóa biên – trường hợp này lệnh BUY (nếu có) rất khó khớp (vì không ai bán khi giá trần, hoặc không ai mua khi giá sàn). Những lệnh rơi vào tình huống này được đưa vào danh sách watchlist thay vì orders_final.
  * TTL cho lệnh: Nếu lệnh thuộc dạng dừng lỗ (stop) có TTL, thông tin TTL (time-to-live) cũng sẽ được ghi vào orders_quality hoặc một file riêng (hoặc cột trong orders_final nếu cần nhập).
- orders_analysis.txt: file văn bản tóm tắt phân tích, bao gồm: tóm tắt MarketRegime (như đã đề cập), tổng quan số lệnh mua/bán, các danh sách mã đặc biệt (neutral partial, override), cảnh báo (warnings) nếu có, v.v. Đây là nơi người vận hành có thể đọc nhanh hiểu kết quả.
- orders_print.txt: có thể là định dạng text tương tự orders_final nhưng trình bày đẹp để dễ nhìn trong console.
- orders_watchlist.csv: liệt kê các mã BUY bị đẩy vào watchlist do vi phạm yếu tố vi mô hoặc vượt hạn mức trong phiên này. Như đã nói, đó có thể là những mã có điểm cao nhưng bị guard filter loại, hoặc mã giá trần, thanh khoản kém, hoặc vượt new_max. Watchlist giúp nhà đầu tư biết mã nào đáng chú ý mặc dù hệ thống chưa mua – có thể cân nhắc thủ công hoặc chờ phiên sau.
- portfolio_evaluation.txt/.csv: (nếu có) chứa đánh giá danh mục sau khi thực hiện các lệnh – phân bổ theo ngành, chỉ số tập trung (HHI, top-N), thanh khoản tổng, beta danh mục, v.v., giúp người dùng thấy bức tranh rủi ro/lợi suất của danh mục mới.

Cuối cùng, orders_final.csv là đầu ra chính: chứa danh sách lệnh đề xuất cuối cùng, gồm 4 cột Ticker, Side, Quantity, LimitPrice, với các lệnh BUY liệt kê trước, sau đó đến SELL【3†L35-L40】. Đây là file để người dùng sử dụng nhập thẳng vào hệ thống giao dịch (sau khi kiểm tra). Bên cạnh đó, nếu một lệnh BUY nào bị loại vào watchlist, nhà đầu tư có thể tìm trong orders_watchlist.csv lý do và có thể chủ động đưa vào nếu chấp nhận rủi ro.

Như vậy, qua các bước tính toán và kiểm soát, engine đảm bảo các lệnh đưa ra vừa dựa trên phân tích định lượng chi tiết, vừa thỏa mãn các nguyên tắc quản trị rủi ro được đặt ra trong policy.

Tương Tác Ngoại Vi & Tích Hợp Hệ Thống

Phần này mô tả cách Broker GPT tương tác với môi trường bên ngoài và tích hợp vào luồng làm việc của người dùng, bao gồm: nguồn dữ liệu, API server backend, quy trình bất đồng bộ qua GitHub Actions (worker), và frontend (ví dụ extension trình duyệt).

Nguồn Dữ liệu Ngoài

Broker GPT không hoạt động độc lập mà cần dữ liệu thị trường từ các nguồn ngoài:
- API thị trường (VNDirect): Hệ thống sử dụng API của VNDirect (dịch vụ DChart) để tải dữ liệu lịch sử giá cổ phiếu (OHLC) cho hàng trăm mã. Đây là nguồn dữ liệu chủ đạo cho pipeline giá quá khứ【8†L18-L20】. Các lời gọi API thực hiện qua HTTP requests, và engine có cơ chế đảm bảo không tải trùng lặp dữ liệu đã có (cache cục bộ).
- Dữ liệu giá realtime/intraday: Với phiên bản hiện tại, engine cũng có khả năng lấy snapshot giá mới nhất trong phiên (nếu chạy trong giờ giao dịch). Điều này có thể thực hiện qua API (nếu có API realtime) hoặc scraping web. Repository có module scripts/collect_intraday.py – có thể dùng API hoặc qua web; chi tiết phụ thuộc vào việc config. Nguồn phổ biến có thể vẫn từ VNDirect (API streaming) hoặc nguồn thứ ba.
- Dữ liệu ngành: File tĩnh data/industry_map.csv cung cấp mã -> ngành, được cập nhật thủ công. Người dùng cần đảm bảo file này đầy đủ để engine biết ngành của các mã mới. Nếu thiếu mã sẽ cảnh báo.
- Dữ liệu cơ bản (fundamentals): Hệ thống hỗ trợ tích hợp dữ liệu cơ bản từ Vietstock. Có script scripts/collect_vietstock_fundamentals.py sử dụng Playwright (trình duyệt tự động) để thu thập chỉ số P/E, ROE, v.v. cho các mã và lưu vào data/fundamentals_vietstock.csv. Đây không phải bước chạy mỗi phiên, mà người dùng có thể định kỳ chạy broker.sh fundamentals để cập nhật kho dữ liệu cơ bản. Khi file này có, pipeline sẽ tự động merge vào metrics.
- Dữ liệu vĩ mô: Nếu có, data/global_factors.csv do người dùng chuẩn bị (hoặc nguồn khác) sẽ được sử dụng. Hiện tại repo không có script tự động cho phần này, giả định người dùng cập nhật.
- Các thông số khác: Một số chỉ số đặc thù (như EPU, chỉ số USD, giá dầu...) có thể cần người dùng nhập hoặc cập nhật thủ công trong global_factors hoặc config policy.

Nhìn chung, để chạy tốt, hệ thống yêu cầu kết nối mạng để gọi API và cần các file dữ liệu tĩnh (industry_map, có thể fundamentals) được chuẩn bị. Các lỗi thường gặp như thiếu dữ liệu đều được engine phát hiện sớm (ví dụ thiếu lịch sử sẽ dừng và báo để chạy lại cho đủ dữ liệu).

Backend API Server (Flask) và Policy Scheduler

Broker GPT cung cấp một API server (Flask) chạy cục bộ nhằm hỗ trợ tích hợp với frontend (ví dụ một extension trình duyệt Chrome hoặc ứng dụng UI). Mã server nằm trong scripts/api/server.py. Khi khởi động bằng lệnh ./broker.sh server, ứng dụng Flask sẽ chạy trên cổng mặc định 8787 (có thể cấu hình qua biến môi trường PORT)【38†L155-L161】.

Các endpoint chính mà server cung cấp (HTTP REST API) gồm:
- GET /health: Kiểm tra nhanh tình trạng server (trả về {"status": "ok", "ts": <timestamp>} nếu server sống)【13†L416-L424】.
- POST /portfolio/reset: Xóa toàn bộ các file CSV trong thư mục in/portfolio/. Endpoint này dùng để bắt đầu một phiên upload danh mục mới (ví dụ user muốn tải một danh mục khác). Server sẽ dọn thư mục input và reset trạng thái chạy (xóa dấu vết run trước)【13†L309-L318】【13†L426-L434】.
- POST /portfolio/upload: Tiếp nhận nội dung của một file danh mục từ phía frontend. Request gửi lên dạng JSON có {"name": "<filename>", "content": "<CSV content>"}. Server sẽ lấy nội dung base64/UTF-8 đó, tạo file CSV tương ứng trong in/portfolio/ (tên theo <name>.csv). Đồng thời, server cũng ghi file này vào thư mục runs/<stamp>/portfolio/ hiện tại【13†L322-L330】【13†L331-L339】. Mỗi phiên upload có một timestamp duy nhất (_CURRENT_STAMP) để nhóm các file portfolio chung đợt với nhau. Nếu _CURRENT_STAMP chưa có (lần upload đầu tiên sau reset), server tạo một timestamp mới và folder runs/<stamp>/portfolio/ để chứa file【10†L33-L41】【10†L35-L38】. API trả về JSON xác nhận đã lưu file (đường dẫn file trong in/portfolio và runs/..., dung lượng bytes, và pending_commit=True, kèm run_stamp hiện tại)【13†L331-L339】【13†L338-L344】. Lưu ý: tại bước upload, server chưa chạy engine ngay và cũng chưa commit lên Git. Nó chỉ chuẩn bị dữ liệu và chờ yêu cầu kết thúc.
- POST /done: Thông báo rằng quá trình upload danh mục đã hoàn tất và yêu cầu hệ thống thực thi lệnh. Khi nhận /done, server sẽ:
  * Kiểm tra nếu có session đang hoạt động (_CURRENT_STAMP tồn tại). Nếu chưa upload gì mà gọi /done, trả về lỗi.
  * Tiến hành gọi hàm finalize_and_run(): hàm này sẽ liệt kê các file CSV trong thư mục runs/<stamp>/portfolio, nếu có file sẽ thực hiện commit và push tất cả những file đó lên repo Git (kèm thông điệp "runs: add portfolio batch for <stamp>")【13†L389-L397】【13†L399-L407】. Việc commit dùng git commands thông qua hàm _git_commit_push() (đảm bảo cấu hình tên user bot, v.v.)【10†L41-L50】【10†L51-L59】.
  * Sau khi push thành công, server reset _CURRENT_STAMP = None để lần upload sau sẽ tạo stamp mới【13†L398-L406】.
  * Gửi phản hồi JSON gồm: "status": "ok" nếu commit/push thành công (hoặc "error" nếu có lỗi), danh sách file đã commit (committed: [...]), run_stamp vừa xử lý, và trạng thái của policy scheduler nếu có【13†L399-L407】.
  Quan trọng: Server không trực tiếp chạy engine khi nhận /done, thay vào đó ủy thác cho cơ chế CI/CD qua git (xem phần tiếp theo). Do đó, response của /done không chứa ngay kết quả lệnh, mà chỉ xác nhận job đã được gửi đi.

- GET /policy/status: kiểm tra trạng thái Policy Scheduler (xem bên dưới). Nếu auto-run policy tắt, trả về {"auto_run": false}. Nếu bật, trả về {"auto_run": true, "scheduler": {...}} trong đó scheduler có thể bao gồm thông tin lần chạy AI gần nhất, lần kế tiếp, v.v. (phục vụ debug/trạng thái)【13†L419-L427】.

Server cũng hỗ trợ request OPTIONS cho các endpoint trên (phục vụ CORS preflight).

Policy Scheduler: Bên trong server, có cơ chế tự động định kỳ chạy cập nhật policy override bằng AI. Lớp PolicyScheduler chạy trong một thread riêng (daemon) nếu biến môi trường BROKER_POLICY_AUTORUN bật (mặc định bật)【13†L280-L289】. Scheduler được khởi tạo với:
- Một danh sách thời điểm trong ngày (mặc định 10 mốc giờ trong phiên, ví dụ 09:10, 09:40, 10:10,... đến 14:15)【10†L91-L99】【10†L93-L98】.
- Tham số lead (mặc định 10 phút) – nghĩa là trước mỗi mốc giờ 10 phút scheduler sẽ khởi động job để đến đúng giờ đó có kết quả.
- Tham số max_age (mặc định 25 phút) – nghĩa là nếu vì lý do gì quá 25 phút chưa có bản cập nhật policy mới, scheduler sẽ cưỡng bức chạy một lần.

Scheduler hoạt động vòng lặp: tính toán thời điểm chạy kế tiếp, ngủ chờ đến gần thời điểm đó, rồi kích hoạt job. Mỗi job chạy sẽ gọi lệnh bash broker.sh policy thông qua hàm run_cmd(['bash', 'broker.sh', 'policy'])【19†L213-L221】. Lệnh này thực chất chạy script scripts/ai/generate_policy_overrides.py (như đã phân tích ở phần Policy) để sinh policy_overrides mới bằng AI. Kết quả job (thành công hay thất bại, thời gian bắt đầu/kết thúc, trigger) được lưu vào đối tượng PolicyRunRecord và scheduler tính lịch chạy tiếp【19†L219-L227】【19†L235-L243】. Trong quá trình chạy, server log ra console các thông báo “[srv] === policy job start [trigger]...” và “done” kèm ok=true/false【19†L214-L222】【19†L229-L237】.

Scheduler cũng cung cấp hàm ensure_ready() – có thể được gọi để đảm bảo có một policy override mới nhất (nếu quá cũ thì chạy ngay một job). Endpoint server có thể gọi cái này nếu cần (hiện tại chưa có endpoint công khai, nhưng có thể được dùng nội bộ).

Tóm lại, PolicyScheduler tự động commit những thay đổi policy override (qua broker.sh policy có sẵn logic commit & push【38†L122-L130】【38†L101-L109】) lên repo theo lịch định trước trong phiên. Điều này cho phép hệ thống luôn có config/policy_overrides.json mới nhất (AI cập nhật vài lần trong phiên nếu cần). Mặc định khi chạy server, auto-run bật, do đó user không cần thủ công gọi AI mỗi ngày. Nếu muốn tắt tính năng này, có thể đặt BROKER_POLICY_AUTORUN=0 khi chạy server.

Xử Lý Bất Đồng Bộ qua GitHub Actions (Worker và Queue)

Khi server nhận yêu cầu /done, nó commit dữ liệu danh mục lên Git repository. Ở phía cloud, repository minhhai2209/broker-gpt-2 được cấu hình một GitHub Actions workflow đặc biệt để xử lý các commit này như một job tính toán.

File workflow .github/workflows/runs-orders.yml định nghĩa việc này:
- Workflow được kích hoạt (on: push) khi có bất kỳ commit nào vào đường dẫn runs/**/portfolio/**【27†L2-L10】. Tức là mỗi khi server push một folder runs/<timestamp>/portfolio/...csv, workflow sẽ chạy.
- Workflow chạy trên máy ảo Ubuntu (GitHub runner) – đóng vai trò như worker thực thi engine.
- Bước đầu, workflow checkout code và tìm các folder runs/<stamp> chưa được xử lý (pending stamps). Nó đánh dấu một stamp “đã xử lý” nếu thấy trong folder đó đã có kết quả (file orders_final.csv). Nếu chưa có kết quả thì liệt kê vào danh sách pending để xử lý【27†L22-L30】【27†L31-L39】【27†L41-L48】.
- Sau đó, nếu có stamp pending, workflow thiết lập môi trường Python (3.11), cài các dependencies (pip install -r requirements.txt)【27†L65-L73】【27†L74-L78】.
- Tiếp theo, bước “Run orders for pending stamps”: script shell sẽ loop qua từng stamp cần xử lý:
  * Dọn dẹp thư mục in/portfolio local và copy mọi file CSV từ runs/<stamp>/portfolio/ vào in/portfolio/【27†L93-L100】 (giúp engine đọc danh mục như bình thường).
  * Xóa thư mục out/orders cũ nếu có (đảm bảo kết quả sạch)【27†L99-L101】.
  * Chạy lệnh ./broker.sh orders – tức chạy toàn bộ pipeline và order engine trên danh mục vừa copy【27†L99-L102】.
  * Sau khi chạy xong, kiểm tra phải có file out/orders/orders_final.csv. Nếu thiếu, coi như job fail (có thể engine lỗi)【27†L103-L107】.
  * Nếu có kết quả, tạo thư mục đích runs/<stamp> (đã có) và copy toàn bộ file trong out/orders/ vào thư mục runs/<stamp>/【27†L107-L112】. Như vậy, kết quả lệnh (orders_final.csv, orders_watchlist.csv, orders_quality.csv, orders_reasoning.csv, orders_analysis.txt, ...) đều được đặt vào thư mục runs tương ứng với đợt đó.
- Bước cuối, workflow commit và push các kết quả này lên repo:
  * Thêm tất cả thay đổi trong thư mục runs/<stamp> vào git, commit với thông điệp "runs: results for <stamp>"【27†L128-L136】【27†L138-L142】, rồi push lên nhánh hiện tại.
  * (Workflow có thiết lập concurrency để không chạy song song hai job trên cùng một nhánh, tránh xung đột khi commit【27†L7-L10】).

Kết quả là, sau vài phút từ lúc người dùng gọi /done, repository sẽ có một commit mới chứa các file kết quả trong runs/<stamp>/. Nhờ đó, phía người dùng có thể truy cập kết quả thông qua repo:
- Server có thể (tương lai) poll hoặc nhận webhook để biết khi nào kết quả xong. Hiện tại, kiến trúc có thể yêu cầu phía frontend trực tiếp kiểm tra kết quả qua GitHub API hoặc tải raw file. (Ví dụ extension có thể sử dụng GitHub API key của user để fetch nội dung file runs/<stamp>/orders_final.csv khi có thông báo hoàn tất).
- Trong response của /done, server trả về run_stamp và thông tin commit thành công. Frontend có thể dùng run_stamp này để biết thư mục kết quả trên repo.

Như vậy, GitHub Actions ở đây đóng vai trò như worker hàng đợi: mỗi phiên chạy (một batch danh mục) tương ứng một commit đẩy vào hàng, runner pick up và xử lý. Kiến trúc này tuy độc đáo (dựa trên git) nhưng có lợi: tận dụng được tài nguyên cloud CI, giảm tải cho máy cục bộ của người dùng, và tách bạch quá trình tính toán nặng khỏi giao diện.

Chú ý bảo mật: Quá trình trên dùng repository của user (hoặc fork) để lưu dữ liệu danh mục và kết quả. Do đó, dữ liệu có thể ở chế độ riêng tư nếu repo private, hoặc công khai nếu repo public. Người dùng cần cân nhắc điều này (ví dụ repo private để bảo mật danh mục). Phía server có sử dụng GitHub CLI (gh) để commit/push; user cần cấu hình quyền truy cập git phù hợp.

Frontend (Extension) và Luồng Tương Tác Người Dùng

Kiến trúc hệ thống cho phép tích hợp với một frontend giúp người dùng không cần thao tác trong terminal. README có đề cập tới một browser extension kết nối với API server. Chức năng của extension có thể là:
- Cho phép người dùng chọn hoặc nhập danh mục ngay trên trình duyệt (có UI để upload file CSV hoặc nhập mã cổ phiếu).
- Gửi danh mục đến server (gọi API /portfolio/upload cho từng file hoặc toàn bộ danh mục).
- Sau khi upload xong, người dùng bấm “Done” trên extension, extension sẽ gọi API /done để bắt đầu tính toán.
- Extension sau đó có thể hiển thị trạng thái “Đang tính toán…” trong thời gian đợi GitHub Actions chạy (~ vài phút).
- Khi nhận được thông báo hoàn thành (có thể extension tự kiểm tra trạng thái repo hoặc server có thể được cấu hình gửi một tín hiệu), extension sẽ tải về file kết quả (ví dụ orders_final.csv) và hiển thị dưới dạng danh sách lệnh trên giao diện cho người dùng xem.
- Ngoài ra, extension có thể dùng các endpoint khác: ví dụ /portfolio/reset để xóa danh mục cũ, hoặc hiển thị nhanh /health để kiểm tra server chạy chưa.

Hiện tại, server chưa có endpoint trả kết quả trực tiếp, nên nhiều khả năng extension sẽ truy cập thẳng các file trong thư mục runs/<stamp> trên máy cục bộ hoặc qua GitHub. Một cách khác: extension có thể được cài trong cùng repo (if permissions allow) và lắng nghe event commit (webhook) – tuy nhiên kiến trúc này phức tạp. Có thể trong tương lai server sẽ có endpoint kiểu /results/<stamp> để trả về nội dung file lệnh. 

Từ góc nhìn kiến trúc:
- Frontend (extension): giao diện người dùng, thu thập input và hiển thị output. Kết nối nhẹ với server qua HTTP (CORS đã được bật trên Flask server để cho phép extension gọi từ localhost hoặc file://).
- Backend (Flask server): nhận yêu cầu từ frontend, quản lý phiên upload, đảm bảo dữ liệu được đưa vào pipeline chính xác, sau đó sử dụng queue (GitHub Actions) để xử lý nền.
- Worker (GitHub Actions): thực thi core engine, tách biệt khỏi frontend. Kết quả trả lại backend thông qua commit code.

Luồng này tạo thành một vòng kín: người dùng thao tác trên UI -> call đến backend -> backend đưa nhiệm vụ vào queue -> queue xử lý xong -> kết quả sẵn sàng -> người dùng nhận kết quả trên UI.

Kết Luận

Tài liệu kiến trúc hệ thống (SYSTEM_DESIGN) này mô tả các thành phần và luồng hoạt động của Broker GPT theo trạng thái codebase hiện tại. Hệ thống gồm engine phân tích & ra quyết định giao dịch theo pipeline dữ liệu và policy, stateless với khả năng audit cao qua các file đầu ra. Cấu trúc modular (pipeline, tính điểm, quyết định lệnh, quản lý trạng thái) tương tác qua data frame và config. Bên ngoài, hệ thống tích hợp API server và cơ chế async qua GitHub Actions để phục vụ trải nghiệm người dùng qua extension/ứng dụng.

Từ danh mục đầu vào, Broker GPT tự động thu thập dữ liệu, đánh giá thị trường, lên chiến lược và đề xuất danh sách lệnh giao dịch, đồng thời vẫn duy trì tính linh hoạt (nhờ AI) và an toàn (guardrails kiểm soát rủi ro) trong mọi điều kiện thị trường.
Calibrations & Execution Diagnostics

Mục này tổng hợp các cơ chế hiệu chỉnh (calibration) và chẩn đoán thực thi (execution diagnostics) được engine sử dụng. Đây là các thành phần kỹ thuật nhằm phản ánh thực trạng thị trường vào tham số vận hành và giúp người dùng hiểu chất lượng lệnh.

- Chi phí giao dịch & slippage
  - Tham số `pricing.tc_roundtrip_frac` biểu diễn chi phí khứ hồi (mua+bán) ở dạng tỷ lệ. Slippage được mô hình hóa tuyến tính theo quy mô lệnh và thanh khoản; các ước lượng được phản ánh trong `orders_quality.csv` qua cột `SlipBps` và `SlipPct`, đồng thời được khấu trừ vào `ExpR` (expected return sau phí).
  - Mục tiêu: tránh giả định “touch là khớp” và không đánh giá quá lạc quan lợi nhuận kỳ vọng khi khối lượng lệnh lớn so với thanh khoản.

- Xác suất khớp & FillRate kỳ vọng
  - Engine ước lượng `FillProb` và `FillRateExp` dựa trên quy mô lệnh tương đối so với thanh khoản (ví dụ ADTV) và các ràng buộc thực thi. Hai chỉ số này hiển thị trong `orders_quality.csv` và dùng để ưu tiên/thả vào watchlist khi khả năng khớp thấp.
  - Khi mã rơi vào trạng thái biên độ khóa (limit‑up/limit‑down), engine gắn cờ `LimitLock` nhằm cảnh báo BUY/SELL khó khớp; các lệnh liên quan có thể bị chuyển vào watchlist.

- TTL cho lệnh và hiệu chỉnh theo biến động thị trường
  - TTL mặc định trong policy: `orders_ui.ttl_minutes.base/soft/hard = 12/9/7` phút. TTL có thể co giãn theo mức biến động thị trường đo bằng ước lượng Garman–Klass trên VNINDEX (cửa sổ gần nhất), nhằm phản ánh trạng thái “bình thường ↔ nhiễu động cao”.
  - Script hỗ trợ: `python scripts/engine/calibrate_ttl_minutes.py`.
    - Input: `out/prices_history.csv` (để tính GK) và `out/orders/policy_overrides.json` hiện tại.
    - Cơ chế: ánh xạ biến động vào 3 bucket `low/medium/high` (ngưỡng có thể cấu hình/ghi trong metadata), sau đó cập nhật `orders_ui.ttl_minutes` theo bucket.
    - Mapping hiện hành: `low → 14/11/8`, `medium → 11/9/7`, `high → 8/6/5` (lần lượt `base/soft/hard`, tính bằng phút).
    - Output: ghi đè vào `config/policy_overrides.json` (hoặc `out/orders/policy_overrides.json` runtime) các khóa liên quan TTL và metadata: `ttl_bucket_minutes`, `ttl_bucket_thresholds`, `ttl_bucket_state`. Lần chạy Order Engine kế tiếp sẽ sử dụng TTL mới.

Lưu ý: Các calibration và diagnostics trên phải được kiểm định bằng dữ liệu khách quan. Khi thay đổi mô hình/slopes/ngưỡng, cập nhật policy, baseline và tests kèm theo để đảm bảo CI xanh và hành vi nhất quán.
