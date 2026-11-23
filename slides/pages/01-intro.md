<div class="section-tag">01 • GIỚI THIỆU BÀI TOÁN</div>

# Bài toán &amp; Mục tiêu Nghiên cứu

<div class="grid grid-cols-12 gap-4 items-start mt-1">
  <div class="col-span-6 space-y-2.5">
    <div class="card">
      <h2>Bối cảnh &amp; Bài toán</h2>
      <ul class="text-xs mb-0">
        <li><strong>Bài toán:</strong> <span class="todo-pill">[TODO: Mô tả ngắn bài toán phân loại, đầu vào X và nhãn đầu ra Y]</span></li>
        <li><strong>Ý nghĩa thực tiễn:</strong> <span class="todo-pill">[TODO: Tại sao bài toán này quan trọng hoặc hữu ích trong thực tế]</span></li>
      </ul>
    </div>
    <div class="card">
      <h2>Mục tiêu của Đồ án</h2>
      <ul class="text-xs mb-0">
        <li>Khám phá dữ liệu (EDA) và thiết kế pipeline tiền xử lý phù hợp.</li>
        <li>Huấn luyện và so sánh ít nhất <strong>2 mô hình phân loại</strong> trên cùng tập kiểm thử.</li>
        <li>Đánh giá bằng Accuracy, Precision, Recall, F1-score và trực quan hóa kết quả.</li>
        <li>Phân tích tác động tiền xử lý, lỗi dự đoán và rút ra bài học môn học.</li>
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
            <div class="text-[11px] text-slate-500">Dataset Kaggle / Roboflow, phân bố lớp &amp; đặc trưng</div>
          </div>
        </div>
        <div class="text-center text-xs text-blue-600 font-bold leading-none">↓</div>
        <div class="flex items-center gap-2.5 rounded-md border border-slate-200 bg-white px-2.5 py-1.5">
          <span class="w-5 h-5 rounded-full bg-blue-700 text-white text-[11px] font-bold flex items-center justify-center shrink-0">2</span>
          <div class="leading-tight">
            <div class="text-xs font-semibold text-slate-900">Tiền xử lý &amp; Augmentation</div>
            <div class="text-[11px] text-slate-500">Chia Stratified Train / Val / Test, chuẩn hóa trên Train</div>
          </div>
        </div>
        <div class="text-center text-xs text-blue-600 font-bold leading-none">↓</div>
        <div class="flex items-center gap-2.5 rounded-md border border-slate-200 bg-white px-2.5 py-1.5">
          <span class="w-5 h-5 rounded-full bg-blue-700 text-white text-[11px] font-bold flex items-center justify-center shrink-0">3</span>
          <div class="leading-tight">
            <div class="text-xs font-semibold text-slate-900">Huấn luyện 2+ Mô hình</div>
            <div class="text-[11px] text-slate-500">Baseline vs. Deep Learning &amp; tinh chỉnh siêu tham số</div>
          </div>
        </div>
        <div class="text-center text-xs text-blue-600 font-bold leading-none">↓</div>
        <div class="flex items-center gap-2.5 rounded-md border border-slate-200 bg-white px-2.5 py-1.5">
          <span class="w-5 h-5 rounded-full bg-blue-700 text-white text-[11px] font-bold flex items-center justify-center shrink-0">4</span>
          <div class="leading-tight">
            <div class="text-xs font-semibold text-slate-900">Đánh giá &amp; Trực quan hóa</div>
            <div class="text-[11px] text-slate-500">Accuracy, Precision, Recall, F1, Loss &amp; Confusion Matrix</div>
          </div>
        </div>
        <div class="text-center text-xs text-blue-600 font-bold leading-none">↓</div>
        <div class="flex items-center gap-2.5 rounded-md border border-slate-200 bg-white px-2.5 py-1.5">
          <span class="w-5 h-5 rounded-full bg-blue-700 text-white text-[11px] font-bold flex items-center justify-center shrink-0">5</span>
          <div class="leading-tight">
            <div class="text-xs font-semibold text-slate-900">So sánh, Phân tích Lỗi &amp; Kết luận</div>
            <div class="text-[11px] text-slate-500">Tác động tiền xử lý, thách thức &amp; liên hệ môn học</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!--
[0:15 – 0:50 | ~35s]
- Trước hết, nhóm trình bày bối cảnh bài toán phân loại [TODO: nêu ngắn gọn đầu vào và các lớp mục tiêu].
- Để giải quyết bài toán này, nhóm bám sát quy trình chuẩn của một dự án Machine Learning gồm 5 bước: từ khám phá dữ liệu, tiền xử lý, huấn luyện ít nhất hai mô hình để đối chứng, đánh giá đa chỉ số cho đến phân tích sâu nguyên nhân lỗi và bài học rút ra.
-->
