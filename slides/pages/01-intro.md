<div class="section-tag">01 • GIỚI THIỆU BÀI TOÁN</div>

# Bài toán &amp; Mục tiêu Nghiên cứu

<div class="grid grid-cols-12 gap-4 items-start mt-1">
  <div class="col-span-6 space-y-2">
    <div class="card py-1.5">
      <h2>Bối cảnh &amp; Bài toán</h2>
      <ul class="text-xs mb-0 leading-snug">
        <li><strong>Hồi quy liên tục (Regression):</strong> Dự đoán giá bán nhà (<code>Price</code>, tỷ VND) từ cấu trúc ngôi nhà và địa chỉ tin đăng.</li>
        <li><strong>Ý nghĩa:</strong> Định giá tham khảo minh bạch và phát hiện bất thường giá giữa các khu vực đô thị.</li>
      </ul>
    </div>
    <div class="card py-1.5">
      <h2>Mục tiêu của Đồ án</h2>
      <ul class="text-xs mb-0 leading-snug">
        <li>Khám phá dữ liệu (EDA), xử lý thiếu nặng và chuẩn hóa văn bản tiếng Việt.</li>
        <li>Triển khai <strong>2 hướng chính</strong>: <code>Linear (SGD)</code> &amp; <code>MLP (Adam)</code>, đối chiếu tham chiếu cây <code>HGB / RF</code>.</li>
        <li>Đánh giá bằng <code>RMSE</code>, <code>MAE</code>, <code>R²</code>, <code>MAPE</code> và phân tích phần dư Test.</li>
        <li>Đo tác động Feature Engineering &amp; liên hệ bài học môn học.</li>
      </ul>
    </div>
  </div>

  <div class="col-span-6">
    <div class="card">
      <h2 class="text-center mb-2">Quy trình Xây dựng Dự án ML</h2>
      <div class="space-y-1">
        <div class="flex items-center gap-2.5 rounded-md border border-slate-200 bg-white px-2.5 py-1.5">
          <span class="w-5 h-5 rounded-full bg-blue-700 text-white text-[11px] font-bold flex items-center justify-center shrink-0">1</span>
          <div class="leading-tight">
            <div class="text-xs font-semibold text-slate-900">Thu thập &amp; Khám phá (EDA)</div>
            <div class="text-[11px] text-slate-500">30,229 tin đăng Kaggle, kiểm tra thiếu &amp; ngoại lai</div>
          </div>
        </div>
        <div class="text-center text-xs text-blue-600 font-bold leading-none">↓</div>
        <div class="flex items-center gap-2.5 rounded-md border border-slate-200 bg-white px-2.5 py-1.5">
          <span class="w-5 h-5 rounded-full bg-blue-700 text-white text-[11px] font-bold flex items-center justify-center shrink-0">2</span>
          <div class="leading-tight">
            <div class="text-xs font-semibold text-slate-900">Tiền xử lý &amp; Feature Engineering</div>
            <div class="text-[11px] text-slate-500">Group-Stratified Split, NFC Địa chỉ, OOF Price/m² priors</div>
          </div>
        </div>
        <div class="text-center text-xs text-blue-600 font-bold leading-none">↓</div>
        <div class="flex items-center gap-2.5 rounded-md border border-slate-200 bg-white px-2.5 py-1.5">
          <span class="w-5 h-5 rounded-full bg-blue-700 text-white text-[11px] font-bold flex items-center justify-center shrink-0">3</span>
          <div class="leading-tight">
            <div class="text-xs font-semibold text-slate-900">Huấn luyện &amp; Tối ưu Mô hình</div>
            <div class="text-[11px] text-slate-500">Linear (Mini-batch SGD), MLP (Backprop + Adam), tham chiếu HGB / RF</div>
          </div>
        </div>
        <div class="text-center text-xs text-blue-600 font-bold leading-none">↓</div>
        <div class="flex items-center gap-2.5 rounded-md border border-slate-200 bg-white px-2.5 py-1.5">
          <span class="w-5 h-5 rounded-full bg-blue-700 text-white text-[11px] font-bold flex items-center justify-center shrink-0">4</span>
          <div class="leading-tight">
            <div class="text-xs font-semibold text-slate-900">Đánh giá &amp; Trực quan hóa</div>
            <div class="text-[11px] text-slate-500">RMSE, MAE, R², MAPE, Biểu đồ Thực tế vs. Dự đoán</div>
          </div>
        </div>
        <div class="text-center text-xs text-blue-600 font-bold leading-none">↓</div>
        <div class="flex items-center gap-2.5 rounded-md border border-slate-200 bg-white px-2.5 py-1.5">
          <span class="w-5 h-5 rounded-full bg-blue-700 text-white text-[11px] font-bold flex items-center justify-center shrink-0">5</span>
          <div class="leading-tight">
            <div class="text-xs font-semibold text-slate-900">Phân tích Phần dư &amp; Kết luận</div>
            <div class="text-[11px] text-slate-500">Tác động tiền xử lý, thách thức &amp; bài học môn học</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!--
[0:15 – 0:50 | ~35s]
- Bài toán của nhóm là dự đoán trực tiếp giá bán nhà (đơn vị tỷ VND) từ thông tin diện tích, số tầng, phòng, pháp lý, nội thất và chuỗi địa chỉ tiếng Việt.
- Nhóm triển khai quy trình ML chuẩn 5 bước trên hơn 30 nghìn tin đăng thực tế, chú trọng khâu chống rò rỉ dữ liệu, tạo đặc trưng địa lý/văn bản và đối chiếu công bằng giữa các họ mô hình hồi quy.
-->
