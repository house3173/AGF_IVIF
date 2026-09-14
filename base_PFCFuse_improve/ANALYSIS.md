# Phân tích PFCFuse gốc và hướng cải tiến

## 1. Kết luận và phạm vi

Đề xuất: giữ nguyên pretrained PFCFuse, học hai adapter residual có biên độ giới hạn tại điểm hợp nhất đặc trưng, huấn luyện trên đúng miền AGF và chọn checkpoint bằng validation đối chứng.

Không có cơ sở khoa học để bảo đảm trước rằng một kiến trúc mới sẽ thắng original trên mọi dataset hoặc mọi metric. Bản triển khai này bảo đảm một điều hẹp hơn: khởi tạo tái tạo đúng phép cộng gốc, không làm thay đổi backbone khi huấn luyện, và giữ baseline nếu chưa có ứng viên vượt tiêu chí validation. Validation tốt hơn vẫn không bảo đảm test tốt hơn.

Phân tích dựa trên **mã nguồn hiện có trong repository**, không khẳng định đây là bản tái hiện nguyên vẹn mọi chi tiết của bài báo PFCFuse. Chưa chạy huấn luyện đầy đủ hay xác nhận mức tăng chất lượng; các nguyên nhân giảm chất lượng dưới đây cần phân biệt giữa lỗi xác định được và giả thuyết cần ablation.

## 2. Mô hình original thực sự làm gì?

### 2.1. Encoder dùng chung trọng số

Nguồn: [net.py](../base_PFCFuse_original/net.py).

- VIS và IR đều là tensor `[B, 1, H, W]`, xử lý riêng qua cùng encoder.
- Conv 3x3 tạo 64 kênh, không downsample; độ phân giải đặc trưng giữ nguyên.
- Bốn `TransformerBlock` trích xuất đặc trưng, mỗi block có LayerNorm, MDTA, residual và Gated-Dconv Feed-Forward.
- MDTA chia 8 head. Tích QK là tương quan giữa các kênh trong mỗi head, không phải attention toàn cục với ma trận `(H*W) x (H*W)`.
- Tuy vậy, thống kê attention vẫn phụ thuộc toàn ảnh. Train trên patch và suy luận trên full image có thể khác nhau; validation phải dùng ảnh đầy đủ.

### 2.2. Nhánh base

`BaseFeatureExtraction` là một khối kiểu PoolFormer:

$$
X' = X + \gamma_1[\operatorname{AvgPool}(\operatorname{LN}(X))-\operatorname{LN}(X)]
$$

$$
B = X' + \gamma_2\operatorname{MLP}(\operatorname{LN}(X'))
$$

MLP dùng conv 1x1, mở rộng 4 lần. Layer scale khởi tạo bằng $10^{-5}$, tạo một khối gần identity trước khi học. Base ở đây là một **biểu diễn đặc trưng học được**, không phải bộ lọc thông thấp bảo đảm chỉ chứa tần số thấp.

### 2.3. Nhánh detail

Encoder dùng ba `DetailNode`, chia 64 kênh thành hai phần 32 kênh, trộn bằng conv 1x1 và cập nhật coupling:

$$
Z'_2 = Z_2 + \phi(Z_1),\qquad
Z'_1 = Z_1\exp(\rho(Z'_2)) + \eta(Z'_2).
$$

Các hàm phi/rho/eta là inverted residual CNN với conv pointwise, depthwise và ReLU6. Cấu trúc lấy cảm hứng từ coupling/invertible networks; không nên suy ra toàn mạng chắc chắn khả nghịch chỉ từ tên lớp. Thừa số exponential cũng khiến thay đổi phân phối đầu vào có thể ảnh hưởng mạnh đến biên độ đặc trưng.

### 2.4. Fusion và decoder

Hai modality được cộng trước các lớp fusion:

$$
B_F = P(B_V+B_I),\qquad D_F=N(D_V+D_I).
$$

`P` là một BaseFeatureExtraction; `N` chỉ có một DetailNode, khác với ba node trong encoder. Decoder ghép hai tensor 64 kênh thành 128 kênh, giảm về 64 bằng conv 1x1, chạy bốn TransformerBlock và hai conv đầu ra.

Ảnh VIS được cộng vào **logit** trước sigmoid:

$$
F=\sigma\left(\operatorname{DecoderFeatures}(B_F,D_F)+V\right).
$$

Đây không phải phép cộng ảnh VIS sau sigmoid. Skip connection này tạo bất đối xứng VIS/IR, nên tự ý đổi thứ tự hai ảnh sẽ thay đổi kết quả. Wrapper trong repository trả trực tiếp đầu ra sigmoid; dòng min-max normalization đã bị comment. Bản mới giữ nguyên điều đó.

### 2.5. Huấn luyện gốc

Nguồn: [train.py](../base_PFCFuse_original/train.py), [loss.py](../base_PFCFuse_original/utils/loss.py).

- 120 epoch: 40 epoch tái tạo riêng từng modality, 80 epoch fusion.
- Phase I: SSIM + Huber, gradient VIS, relative mean difference và decomposition loss.
- Decomposition loss: $CC_D^2/(1.01+CC_B)$, khuyến khích base tương đồng và detail ít tương quan.
- Phase II: fusion loss + 2 lần decomposition loss. Các biến SSIM/Huber được tính trong Phase II nhưng không được cộng vào loss cuối.
- Fusion intensity target là max(VIS, IR), gradient target là max độ lớn Sobel.
- Learning rate $10^{-4}$, batch size 8, gradient clipping 0.01. Không có tập validation trong script gốc.

Điểm cần đặc biệt lưu ý trong loss hiện tại: `relative_diff_loss` là sai lệch **trung bình cường độ vô hướng**, không phải mutual information. `Fusionloss` lấy max của hai số vô hướng này rồi gọi `F.l1_loss(scalar, generate_img)`, dẫn tới broadcast scalar lên toàn ảnh. Như vậy mục tiêu thực tế là kéo từng pixel về một đại lượng sai lệch trung bình, không phải bảo toàn tương quan cục bộ. Bản cải tiến không tái sử dụng biểu thức này.

## 3. Vì sao bản CMA + SCAF có thể không hiệu quả?

### 3.1. Các vấn đề xác định được từ code

**Đối chứng có thể bị lẫn kiến trúc.** [Wrapper original](../base_PFCFuse_original/base_PFCFuse.py) import `base_PFCFuse.net`, không import net trong chính thư mục original. Chỉ thay import wrapper ở main chưa đủ bảo đảm đang chạy mạng gốc. Bộ test mới nạp trực tiếp net original bằng đường dẫn độc lập để tránh lỗi này.

**Fallback cộng không đồng nghĩa fallback original.** Trong [wrapper CMA](../base_PFCFuse/base_PFCFuse.py), dù thiếu trọng số SCAF, lời gọi vẫn là `Encoder(vi, ir)` và đi qua CMA. Khi checkpoint gốc thiếu CMA, `strict=False` bỏ qua các trọng số thiếu, giữ CMA khởi tạo ngẫu nhiên. Nếu tải checkpoint đã chứa đầy đủ CMA thì tình huống này không xảy ra, nhưng vẫn phải kiểm tra tính tương thích thay vì im lặng bỏ qua key.

**Scale fusion bị thay đổi.** SCAF trả $w_VX_V+w_IX_I$ với $w_V+w_I=1$. Khi trọng số cân bằng, kết quả gần $(X_V+X_I)/2$, khác phép cộng mà lớp fusion/decoder pretrained đã học. LayerNorm không chứng minh toàn pipeline bất biến với sự thay đổi đó vì còn residual, coupling, conv và sigmoid.

**CMA làm thay đổi đặc trưng pretrained ngay từ đầu.** CrossAttention và FFN mới không có zero-init residual gate. Mạng mới không bắt đầu tại nghiệm pretrained, trong khi training rút xuống 40 epoch và bỏ Phase I.

**Chọn best theo training loss.** Script mới lưu best theo loss train trung bình, chưa đo khả năng tổng quát hóa hoặc chất lượng tái tạo đầy đủ AGF+MLE.

### 3.2. Các giả thuyết hợp lý nhưng chưa phải kết luận thực nghiệm

- Lệch miền: pretrained/train dùng ảnh thô, nhưng [main.py](../main.py) đưa ảnh base đã qua AGF vào encoder.
- Tối ưu quá nhiều thành phần cùng lúc có thể phá kiến thức pretrained trước khi attention mới học được cách dùng đặc trưng.
- CMA có thể khiến hai modality giống nhau hơn, đối nghịch một phần với mục tiêu giữ thông tin detail riêng biệt. Cần đo CC và ablation để kết luận.
- Bảo toàn max gradient trên ảnh thô không nhất thiết giúp base fusion: WLE/MLE đã cung cấp chi tiết, nên tăng gradient ở base có thể tạo dư chi tiết hoặc halo sau cộng.
- Metric như entropy hoặc spatial frequency có thể tăng khi nhiễu tăng. Không được coi một metric tăng là bằng chứng ảnh tốt hơn toàn diện.

## 4. Thiết kế đã triển khai

Nguồn: [fusion_model.py](fusion_model.py).

Với mỗi nhánh base/detail, đặt:

$$
X_F=X_V+X_I+g(X_V,X_I)\odot(X_V-X_I),\quad
g=\alpha\tanh(h_\theta(X_V,X_I)),\quad \alpha=0.25.
$$

Tương đương:

$$
X_F=(1+g)\odot X_V+(1-g)\odot X_I.
$$

Mỗi hệ số nằm trong [0.75, 1.25], tổng hai hệ số luôn bằng 2. Đây là redistribution quanh phép cộng, không phải thay phép cộng bằng trung bình.

Adapter nhận đặc trưng hai modality, độ khác biệt tuyệt đối và ngữ cảnh average-pooling 5x5. Chuẩn hóa RMS cục bộ chỉ áp dụng cho đầu vào adapter, không thay đổi feature backbone. Conv 1x1 nén xuống 24 kênh (v2, trước là 16), ba nhánh depthwise conv 3x3 dilation 1/2/4 chạy song song rồi cộng lại để mở rộng receptive field (v1 chỉ có dilation 1/2), conv cuối trả gate theo **từng kênh và vị trí**.

Conv cuối khởi tạo weight/bias bằng 0. Do đó:

1. Lúc đầu $g=0$, đầu ra bằng phép cộng gốc chính xác.
2. Khi $X_V=X_I$, residual bằng 0 kể cả sau khi học.
3. $|X_F-(X_V+X_I)|\leq0.25|X_V-X_I|$ theo từng phần tử.
4. Backbone được strict-load và đóng băng hoàn toàn theo mặc định. Gradient vẫn truyền qua các lớp fusion/decoder để học adapter; không bọc decoder candidate trong `no_grad`.

Backbone có 528.544 tham số; hai adapter v2 có tổng khoảng 17.000 tham số (tăng so với 11.040 của v1 do mở rộng hidden width và thêm nhánh dilation 4), tổng model ~545.500. Vẫn dưới 3.3% số tham số backbone; adapter vẫn xử lý feature full-resolution nên chi phí latency/VRAM không tỉ lệ tuyến tính với số tham số.

### 4.1. Tuỳ chọn fine-tune có kiểm soát lớp fusion gốc (`--finetune-fusion-lr`)

Một giả thuyết ở §3.2 là `BaseFuseLayer`/`DetailFuseLayer` (PoolFormer/DetailNode) được train nguyên bản dưới giả định fusion là **phép cộng thuần túy**; residual adapter chỉ điều chỉnh đầu vào của hai lớp này chứ không cho phép chúng thích nghi lại. [train_residual.py](train_residual.py) có tuỳ chọn `--finetune-fusion-lr LR` (mặc định 0, tắt): khi bật, `BaseFuseLayer`/`DetailFuseLayer` được mở khoá gradient và tối ưu ở một nhóm tham số riêng với LR thấp hơn LR của adapter; `DIDF_Encoder`/`DIDF_Decoder` (phần sâu, rủi ro catastrophic forgetting cao hơn) vẫn luôn đóng băng bất kể cờ này. Trước bước cập nhật đầu tiên, hai lớp này vẫn giữ đúng trọng số pretrained nên tính chất "epoch -1 = baseline" không đổi; toàn bộ epoch sau đó vẫn phải qua đúng validation gate như adapter-only mới được chọn — đây không phải cách nới lỏng tiêu chí chấp nhận, chỉ là mở rộng không gian tham số có thể học.

## 5. Loss và miền dữ liệu

Nguồn: [data.py](data.py), [objective.py](objective.py).

Chế độ mặc định dùng AGF cùng tham số của main: sigma_spatial=5, sigma_range=0.05, iterations=4, median_ksize=3, numba=True. Detail fusion mặc định là WLE trong [MVA_WLE.py](../detail_fusion/MVA_WLE.py), dùng cửa sổ 5x5 và reflective padding; MLE vẫn là lựa chọn đối chứng qua `--detail-fusion mle`. Cache giữ VIS, IR, hai base và detail đã fusion; AGF/WLE hoặc AGF/MLE chạy trên full image trước khi crop để không tạo biên giả của bộ lọc tại mép patch.

Mạng sinh base $B_F$, loss được tính trên ảnh sau tái tạo $F=\operatorname{clip}(B_F+D_{WLE},0,1)$ mặc định:

$$
L=L_{int}+L_{ssim}+6L_{grad}+2L_{energy}+0.1L_{low}+L_{trust}+0.2L_{clip}.
$$

- `L_int`: L1 với tổ hợp VIS/IR theo độ tương phản cục bộ, trọng số mỗi modality bị chặn trong [0.2, 0.8]. Đây là prior không giám sát, không phải ground truth hay detector vật thể nóng.
- `L_ssim`: trung bình SSIM loss với hai ảnh nguồn, dùng Kornia.
- `L_grad`: khớp vector Sobel **trên ảnh gốc, không blur** từ modality có năng lượng gradient mạnh hơn.
- `L_energy`: hinge một chiều $\operatorname{relu}(E_{target}-E_{fused})$, chỉ phạt khi năng lượng gradient của ảnh fused thấp hơn nguồn mạnh hơn; không ép tăng tùy tiện.
- `L_low`: L1 giữa các ảnh đã average-pool 11x11, chỉ kiểm soát nội dung tần số rất thấp.
- `L_trust`: chỉ phạt sai khác so với base teacher vượt 0.02, hạn chế phá nghiệm gốc nhưng vẫn cho phép điều chỉnh nhỏ.
- `L_clip`: phạt năng lượng vượt [0,1] trước clipping; tránh coi mất chi tiết do clipping là miễn phí.

Không thêm decomposition loss vì encoder đã bị đóng băng: loss chỉ phụ thuộc feature encoder sẽ là hằng số theo tham số adapter. Các hệ số loss là giá trị khởi đầu hợp lý, chưa được tối ưu thực nghiệm. Một prior có thể sai ở cảnh đêm/nhiễu; đây là lý do phải dùng validation và kiểm tra bằng mắt.

### 5.1. Sự cố phát hiện qua log huấn luyện thật (v1 → v2)

Chạy training thật trên MSRS (20 epoch dự kiến, dừng quan sát ở epoch 7) với objective v1 (bản mô tả ở trên trước khi sửa: `L=L_int+2L_ssim+5L_grad(blur)+0.5L_low+L_trust+L_clip`) cho kết quả: proxy loss giảm đều (0.455→0.393), SSIM tăng (0.682→0.701), clipping giảm mạnh (0.0068→0.0012), nhưng **Qabf không epoch nào vượt baseline** (luôn ~0.659-0.661 so với 0.6694), còn MI giảm từ 3.89 xuống ~3.0-3.14 bit, EN giảm từ 6.79 xuống ~6.45-6.48, SF giảm từ 11.94 xuống ~10.7-10.8.

Đây là **proxy objective bị đánh lừa (reward hacking)**: model tìm được lối tắt rẻ tiền là thu hẹp dải động/độ tương phản ảnh, làm giảm đồng thời entropy, spatial frequency, mutual information và Qabf (đều đo thông tin/độ sắc nét), trong khi vẫn làm giảm đúng các loss đang được tối ưu trực tiếp (SSIM tăng, clipping giảm). Hai nguyên nhân kỹ thuật xác định được:

1. `L_grad` v1 làm blur cả fused và target trước khi tính Sobel. Vì hai phía bị làm mượt như nhau, model có thể xóa chi tiết tần số cao thật (ảnh hưởng SF/EN/MI/Qabf) mà loss này không phát hiện được, do phép so sánh chỉ diễn ra ở thang đã làm mượt.
2. `L_low` (cửa sổ 5x5, trọng số 0.5) và `L_clip` (trọng số 1) cùng kéo ảnh về một target trung bình hóa/dải giá trị hẹp, không có thành phần nào chống lại việc giảm phương sai cục bộ.

Bản sửa (v2, đang dùng trong [objective.py](objective.py)):

- Bỏ blur trong `L_grad`: tính Sobel trực tiếp trên ảnh gốc, tăng trọng số lên 6 để đây là tín hiệu chính giữ biên/chi tiết.
- Thêm `L_energy`: hinge một chiều theo năng lượng gradient trung bình toàn ảnh, chỉ trừng phạt khi fused có ít năng lượng cạnh hơn nguồn mạnh nhất — không thể thỏa mãn bằng cách làm phẳng ảnh.
- Giảm mạnh trọng số `L_low` (0.5→0.1) và mở cửa sổ pooling (5→11) để chỉ ràng buộc tần số rất thấp, không siết local mean ở quy mô trung bình.
- Giảm trọng số `L_clip` (1→0.2) để nó chỉ là lưới an toàn, không phải động lực chính.

Kiểm chứng nhanh (không phải benchmark đầy đủ): với ảnh giả lập, ứng viên phẳng hoàn toàn (`0.5` mọi pixel, đúng kiểu lối tắt log cho thấy) nay có loss **1.819**, cao hơn ứng viên giữ biên độ (`max(VIS,IR)`, loss **0.911**) — với objective v1, loss không phân biệt được hai trường hợp này theo hướng đúng. Đây là bằng chứng thiết kế đã sửa đúng cơ chế gây lỗi, **không phải bằng chứng Qabf sẽ tăng khi train lại**; cần chạy lại training thật trên GPU và evaluate lại để xác nhận.

### 5.2. Vòng lặp thứ hai: từ heuristic tuyệt đối sang ranking tự tham chiếu (v2 → v3)

Sau khi sửa v2, người dùng chạy training thật và báo cáo: các chỉ số **có tốt hơn** so với bản đầu (không còn sụp giảm MI/EN/SF như v1), nhưng **vẫn chưa đủ để Qabf vượt baseline**. Nguyên nhân cấu trúc còn lại của v2: các trọng số `L_int`, `L_ssim`, `L_grad`, `L_low`, `L_clip` là **heuristic tuyệt đối**, được chọn thủ công và chỉ *hy vọng* tương quan với đúng 4 chỉ số mà gate validation kiểm tra (Qabf, SSIM, SCD, clipping). Không có gì trong v2 đảm bảo cực tiểu của tổ hợp loss đó trùng với "Qabf cao hơn baseline, các chỉ số khác không tệ hơn" — đây vẫn là compound risk của việc dùng proxy gián tiếp.

**Chiến lược v3 (đang dùng trong [objective.py](objective.py)):** thay vì tối ưu heuristic tuyệt đối, tính **xấp xỉ khả vi của chính 4 chỉ số trong gate** (Qabf, SSIM, SCD, tỉ lệ clipping) cho **cả candidate lẫn teacher (baseline cộng) trên cùng batch**, rồi tối ưu bằng hinge một chiều so với chính baseline đó:

- `QabfProxy`: cài lại chính xác công thức Xydeas & Petrovic trong [utils/Evaluator.py](utils/Evaluator.py) (cùng kernel Sobel `h1`/`h3`, cùng hằng số `Tg=0.9994, kg=-15, Dg=0.5, Ta=0.9879, ka=-22, Da=0.8`), chỉ thay nhánh rời rạc `arctan`/so sánh bằng `torch.atan` và tỉ số có `eps` ổn định để giữ khả vi. Đây không phải bản thay thế cho Evaluator numpy 8-bit dùng để tính gate/report — nó chỉ là tín hiệu huấn luyện.
- `_scd_score`: hệ số tương quan Pearson khả vi, đúng định nghĩa SCD trong Evaluator ($\operatorname{corr}(A,F-B)+\operatorname{corr}(B,F-A)$).
- SSIM: dùng lại `kornia.losses.ssim_loss` với `reduction="none"` rồi lấy trung bình theo từng ảnh trong batch (không phải trung bình cả batch), để so sánh candidate với teacher **theo từng cặp ảnh**, không lẫn giữa các ảnh khác nhau.
- Clipping: tỉ lệ năng lượng bị cắt trước khi clamp, tính riêng cho candidate và teacher.

Loss cuối cùng gồm 5 số hạng, mỗi số hạng là hinge một chiều theo đúng dung sai của gate tương ứng:

$$
L=w_{Qabf}\operatorname{relu}(m_{Qabf}-(Q_c-Q_t)) + w_{ssim}\operatorname{relu}((S_t-S_c)-m_{ssim}) + w_{scd}\operatorname{relu}((C_t-C_c)-m_{scd}) + w_{clip}\operatorname{relu}((P_c-P_t)-m_{clip}) + L_{trust}
$$

với $Q,S,C,P$ lần lượt là Qabf-proxy, SSIM-proxy, SCD-proxy, tỉ lệ clipping của candidate ($c$) và teacher ($t$); $m_{ssim}=0.001$, $m_{scd}=0.01$, $m_{clip}=0.001$ **lấy đúng bằng dung sai của gate thật** trong [evaluation.py](evaluation.py) (không phải số tuỳ ý mới); $m_{Qabf}=0.01$ là biên độ đòi hỏi vượt trội thật sự (lớn hơn 100 lần ngưỡng tối thiểu 0.0001 của gate, để có margin an toàn khi proxy khả vi không khớp tuyệt đối với Evaluator 8-bit numpy). Trọng số mặc định $w_{Qabf}=4, w_{ssim}=2, w_{scd}=1, w_{clip}=1$, `L_trust` giữ nguyên như v1/v2 (ngưỡng 0.02).

Hệ quả thiết kế quan trọng: tại $candidate=teacher$ (đúng epoch -1 / trước khi học), bốn hinge guard (ssim/scd/clip) đều bằng 0 (không có khoảng cách để phạt), chỉ còn $L=w_{Qabf}\cdot m_{Qabf}$ — một hằng số dương đã được xác minh bằng thực nghiệm (`0.04 = 4.0×0.01`), không phải 0. Điều này đúng ý đồ: loss không coi "bằng baseline" là tối ưu, mà luôn đẩy về phía "vượt baseline".

Kiểm chứng đã chạy (không phải bằng chứng Qabf thật sẽ tăng, chỉ xác nhận thiết kế đúng hướng):

- Ứng viên phẳng (`0.5` mọi pixel) có loss **2.94**, ứng viên sắc nét (`max(VIS,IR)`) dùng làm chính teacher có loss đúng bằng hằng số kỳ vọng **0.04** — chênh lệch rất lớn theo đúng hướng mong muốn.
- `QabfProxy` một mình đã xếp hạng đúng: sharp = 0.301 > flat = 0.070, trên cùng cặp nguồn ngẫu nhiên.
- Toàn bộ 9 unit test (bao gồm test CLI train/evaluate/infer với `--finetune-fusion-lr` và không có cờ này) đều pass sau khi vá một lỗi NaN backward (đạo hàm `sqrt` tại 0 khi ảnh hằng — đã thêm `eps` **vào trong** căn thay vì cộng sau).

**Vẫn chưa có bằng chứng huấn luyện thật với v3** vượt qua baseline; đây là thay đổi thiết kế, không phải kết quả thực nghiệm mới. Điểm khác biệt căn bản so với v1/v2: thay vì đoán trọng số cho một tập heuristic độc lập với gate, v3 tối ưu trực tiếp xấp xỉ của chính các chỉ số gate dùng, trên đúng batch đang huấn luyện — giảm thiểu khoảng cách giữa "cái đang được tối ưu" và "cái sẽ được đánh giá", vốn là nguyên nhân gốc rễ của cả hai lần thất bại trước.

## 6. Quy trình thí nghiệm và tiêu chí chấp nhận

### Chia dữ liệu

- Chia ở cấp ảnh trước khi lấy patch, không chia ngẫu nhiên patch cùng ảnh sang cả train/val.
- Khuyến nghị dùng danh sách validation theo scene/sequence. Random split theo ảnh chỉ là mặc định khi chưa có metadata cảnh, không loại trừ leakage giữa frame lân cận.
- Không dùng MSRS test hoặc TNO để chọn epoch/hyperparameter. Sau khi chốt cấu hình, đánh giá test một lần; nếu tiếp tục điều chỉnh thì test đó không còn độc lập.
- Cache lưu hash cặp tệp; evaluator từ chối cặp test byte-identical với train/val. Hash không phát hiện mọi ảnh gần trùng, ảnh được encode lại hoặc frame cùng cảnh.

### Chọn checkpoint

Mỗi epoch được đánh giá trên toàn bộ ảnh validation nguyên kích thước. Ứng viên phải đồng thời:

1. Tăng Qabf trung bình hơn baseline ít nhất 0.0001.
2. SSIM trung bình không giảm quá 0.001.
3. SCD trung bình không giảm quá 0.01.
4. Tỷ lệ clipping không tăng quá 0.001 (0,1 điểm phần trăm).
5. Proxy loss validation thấp hơn baseline.

Trong các ứng viên đạt, chọn Qabf cao nhất. Các ngưỡng là chính sách mặc định đã công khai, không phải định luật chất lượng; cần chốt trước thí nghiệm. Nếu không có ứng viên đạt, `selected.pth` vẫn là model zero-residual, epoch=-1. `last.pth` có thể kém hơn baseline và không được tự động triển khai.

### Đánh giá test

Evaluator xuất ảnh hai phương pháp, CSV từng ảnh, mean và bootstrap CI 95% của **chênh lệch theo cặp ảnh** cho Qabf, SSIM, SCD, MI, EN, SF, clipping và loss. Miền ảnh đánh giá là PNG 8-bit cùng quy ước xuất ảnh của main; SSIM là trung bình hai nguồn, MI được đổi sang bit. SCD có xử lý tương quan không xác định của ảnh hằng bằng 0.

Nhãn `supported_on_this_split` yêu cầu ít nhất 20 ảnh, thỏa gate trên và cận dưới CI Qabf > 0. Đây là tiêu chí bằng chứng tối thiểu của script, không phải bảo đảm thống kê tuyệt đối. Bootstrap ảnh giả định ảnh độc lập; scene/sequence liên quan cần group bootstrap ngoài script. Chưa điều chỉnh multiple comparisons hoặc bias do thử nhiều cấu hình.

Thiết kế ablation tối thiểu:

| Thí nghiệm | Mục đích |
|---|---|
| Original độc lập, cùng preprocessing | Baseline thật |
| Zero-init residual | Chứng minh không thay output trước khi học |
| `--adapt base` | Vai trò của adapter feature base |
| `--adapt detail` | Vai trò của adapter feature detail |
| `--adapt both` | Hiệu quả phối hợp |
| Ít nhất 3 seed, cùng split | Độ ổn định huấn luyện |
| MSRS test và TNO giữ kín | Khả năng tổng quát hóa |

Ảnh phải được so sánh tại vùng vật thể nhiệt, texture, biên, vùng tối và vùng chuyển độ sáng; cần kiểm tra halo, noise, mất tương phản, clipping. Với mục tiêu detection/segmentation, phải đo mAP/mIoU bằng mô hình downstream cố định; bộ code này chưa triển khai thí nghiệm downstream đó.

## 7. Bằng chứng đã có và giới hạn

- Python 3.12, PyTorch 2.12.0+cpu, Kornia 0.8.3; không có CUDA trong môi trường kiểm thử.
- 8 unittest pass; test CLI được chạy lại sau khi bổ sung xuất ảnh và tiếp tục pass.
- Kiểm tra strict checkpoint, gradient chỉ ở adapter, bound residual, ảnh hằng, input sai, lưu/nạp model, AGF/MLE, train một epoch và chặn leakage.
- Đối chiếu trực tiếp với `base_PFCFuse_original/net.py` bằng checkpoint thật: sai khác tuyệt đối bằng 0 trước khi học.
- Một cặp MSRS `01`, 480x640, chạy AGF+MLE thật: output hữu hạn trong [0,1], residual khởi tạo bằng baseline. Tổng thời gian cho AGF + hai forward trên CPU khoảng 122 giây; không phải benchmark latency inference riêng.
- Chưa huấn luyện đầy đủ, chưa có kết quả test cho model đã học, chưa thử CUDA, chưa đo peak VRAM/latency giữa các kiến trúc. Không có checkpoint nào được tuyên bố chắc chắn tốt hơn original.
- Bản chạy training thật đầu tiên trên GPU (objective v1) cho thấy Qabf không vượt baseline trong 8 epoch đầu, kèm sụt giảm MI/EN/SF — xem §5.1.
- Bản chạy training thật thứ hai (objective v2, người dùng tự chạy) cho kết quả tốt hơn v1 nhưng người dùng báo cáo Qabf vẫn chưa vượt baseline (không có log số liệu đính kèm cho lần chạy này). Đây là động lực cho thiết kế v3 ở §5.2.
- **Chưa có lần chạy training thật nào với v3** (ranking loss tự tham chiếu + adapter receptive field lớn hơn + tuỳ chọn fine-tune fusion layer). Mọi khẳng định về việc v3 vượt baseline đều chưa được kiểm chứng bằng thực nghiệm thật; chỉ có các kiểm chứng đơn vị xác nhận thiết kế phân biệt đúng hướng sharp/flat và đúng hành vi tại điểm khởi tạo.

Chiến lược này có thể thất bại nếu bottleneck nằm ở decoder hoặc MLE chứ không ở fusion. Khi cả ba ablation đều không đạt validation, nên giữ original và phân tích lỗi theo cảnh trước khi mở rộng quyền cập nhật backbone. Không nên tự động thêm nhiều attention hơn hoặc điều chỉnh theo test cho đến khi metric tăng.