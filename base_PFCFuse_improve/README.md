# PFCFuse Improve: bounded residual fusion

Hướng cải tiến giữ nguyên backbone original, chỉ học hai adapter fusion nhỏ, khởi tạo tương đương phép cộng gốc. Đọc [phân tích chi tiết](ANALYSIS.md) để hiểu kiến trúc, loss, lỗi đối chứng và giới hạn bằng chứng.

**Trạng thái:** code đã có train/evaluate/infer và kiểm thử; chưa huấn luyện đầy đủ, chưa chứng minh vượt original. Khi chưa có checkpoint được validation chấp nhận, dùng baseline.

## 1. Các entry point

Chạy các lệnh dưới đây từ thư mục gốc repository, không chạy bên trong thư mục package.

| Tệp                                  | Vai trò                                                  |
| ------------------------------------- | --------------------------------------------------------- |
| [net.py](net.py)                       | Kiến trúc gốc giữ nguyên để nạp pretrained        |
| [fusion_model.py](fusion_model.py)     | Backbone frozen, residual adapters, strict loader         |
| [base_PFCFuse.py](base_PFCFuse.py)     | API thay thế cho main, không import nhầm sang bản CMA |
| [data.py](data.py)                     | Chuẩn bị cặp ảnh, cache AGF và detail fusion WLE/MLE, split train/val/test |
| [train_residual.py](train_residual.py) | Trainer mới và validation gate                          |
| [evaluation.py](evaluation.py)         | Đối chứng, ảnh xuất, CSV, bootstrap CI               |
| [infer.py](infer.py)                   | Xuất ảnh fusion không cần sửa main                   |
| [test_improve.py](test_improve.py)     | Unittest và kiểm thử end-to-end                        |

[train.py](train.py), [dataprocessing.py](dataprocessing.py) và [utils/loss.py](utils/loss.py) là các script legacy được giữ từ bản copy để tham khảo. **Không dùng chúng để huấn luyện cải tiến này**; trainer mới không gọi loss legacy và không yêu cầu HDF5.

## 2. Môi trường

Python **3.8 trở lên**, PyTorch **2.6 trở lên** để dùng restricted checkpoint loading (`weights_only=True`). Đã kiểm thử trên Python 3.12 / Torch 2.12.0 CPU / Kornia 0.8.3, và tương thích cú pháp với Python 3.8.

Khi triển khai sang máy khác (ví dụ server lab), copy cả `base_PFCFuse_original` cùng cấp với `base_PFCFuse_improve`; test đối chứng `test_identity_against_independent_original_net` cần thư mục này để nạp `net.py` gốc độc lập và sẽ tự `skip` (không fail) nếu thiếu.

Các thư viện cần có: torch, numpy, opencv-python, scipy, numba, kornia, timm, einops, scikit-image, scikit-learn. Đây là các thành phần đã có trong repository/môi trường hiện tại; không cần framework mới.

Ví dụ cài các gói ngoài Torch trong môi trường mới:

```powershell
python -m pip install numpy opencv-python scipy numba kornia timm einops scikit-image scikit-learn
```

Cài bản PyTorch CPU/CUDA phù hợp bằng hướng dẫn chính thức của PyTorch. GPU được khuyến nghị cho training đầy đủ; CPU chỉ nên dùng smoke test hoặc suy luận ít ảnh. Giảm batch/crop nếu thiếu VRAM. Chưa xác minh CUDA hoặc dung lượng VRAM tối thiểu trong phiên triển khai này.

Chỉ nạp checkpoint từ nguồn tin cậy. Loader dùng `weights_only=True`, bỏ tiền tố `module.` của DataParallel rồi kiểm tra key/shape bằng `strict=True`. Checkpoint CMA/SCAF không tương thích sẽ bị từ chối, không âm thầm bỏ trọng số thiếu.

## 3. Kiểm thử trước

```powershell
python -m unittest base_PFCFuse_improve.test_improve -v
```

Chín test dùng checkpoint original thật và dữ liệu tổng hợp nhỏ trong thư mục tạm. Kiểm thử bao gồm một epoch, xuất ảnh, đánh giá, fine-tune fusion layer và từ chối test bị trùng. Thời gian thực tế khoảng một phút hoặc hơn trên CPU; kết quả này chứng minh code chạy đúng, không chứng minh chất lượng fusion đã tăng.

## 4. Chọn detail fusion

Mặc định mới là `wle`, gọi `fuse_detail_dynamic_WLE_im2col_var_optimizer()` trong [MVA_WLE.py](../detail_fusion/MVA_WLE.py). Với mỗi detail image sau AGF, nó tính Weighted Local Energy (WLE) theo cửa sổ $5\times5$, trọng số Gaussian thích nghi với variance cục bộ, rồi chọn detail IR hoặc VIS tại pixel có năng lượng lớn hơn:

```python
detail = fuse_detail_dynamic_WLE_im2col_var_optimizer(detail_infrared, detail_visible)
```

MLE cũ vẫn có thể dùng với `--detail-fusion mle`, nhưng **không được trộn** cache/checkpoint WLE với cache/checkpoint MLE. Manifest cache, checkpoint và inference giờ đều lưu/kiểm tra `detail_fusion`; mismatch sẽ dừng với lỗi rõ ràng. Thay WLE không tự đảm bảo metric tăng: nó thay quy tắc chọn chi tiết, do đó phải đánh giá lại toàn bộ pipeline.

## 5. Chuẩn bị train/validation đúng miền AGF + WLE

Nguồn phải có hai thư mục `vi` và `ir`, cặp ảnh trùng stem, cùng kích thước và đã căn chỉnh. Không resize tự động. Mỗi lần chạy cần một thư mục output mới để tránh ghi đè.

```powershell
python -m base_PFCFuse_improve.data --source MSRS/train --output base_PFCFuse_improve/cache/msrs_agf_wle --mode agf --detail-fusion wle --val-fraction 0.15 --seed 3407
```

Script chạy AGF và WLE trên full image, lưu NPZ cùng manifest có nguồn dữ liệu, tham số AGF/WLE, split và hash. Khi train mới crop patch và flip đồng bộ các tensor. Cache làm giảm chi phí AGF/WLE lặp lại mỗi epoch nhưng cần thêm dung lượng đĩa.

Với dữ liệu có scene/sequence, nên chuẩn bị danh sách validation chứa mỗi dòng một image stem và giữ toàn scene trong cùng split. Thêm `--val-list <duong_dan_danh_sach>` khi chuẩn bị cache. Random image split mặc định không ngăn được mọi leakage giữa frame gần nhau.

Không chuẩn bị train từ `MSRS/test` hoặc `data/TNO`. Đối với ảnh thô không dùng AGF/detail fusion, tạo **cache riêng** bằng `--mode raw`; không dùng model học raw thay cho model học agf.

Cache `msrs_agf` cũ được tạo với MLE không thể tái dùng cho WLE: tạo một thư mục cache mới như lệnh trên. Để có đối chứng MLE công bằng, tạo một cache MLE riêng với cùng source, `--val-fraction` và `--seed`:

```powershell
python -m base_PFCFuse_improve.data --source MSRS/train --output base_PFCFuse_improve/cache/msrs_agf_mle --mode agf --detail-fusion mle --val-fraction 0.15 --seed 3407
```

## 6. Huấn luyện

```powershell
python -m base_PFCFuse_improve.train_residual --cache base_PFCFuse_improve/cache/msrs_agf_wle --output base_PFCFuse_improve/runs/msrs_agf_wle_v3_s3407 --epochs 20 --batch-size 4 --crop 128 --lr 0.001 --seed 3407 --device cuda
```

Nếu chỉ có CPU, thay `--device cuda` bằng `--device cpu`, nhưng toàn bộ validation ảnh đầy đủ có thể rất chậm.

Model mặc định nạp checkpoint original trong package; không dùng `best.pth` của bản CMA. Tuỳ chọn `--pretrained` chỉ dành cho checkpoint theo cấu trúc original, không phải checkpoint improve đã học. Các run là độc lập; chưa cung cấp cơ chế resume chính xác toàn bộ RNG/scheduler.

Các thiết lập chính:

- Encoder, decoder luôn đóng băng. BaseFuseLayer/DetailFuseLayer đóng băng theo mặc định, chỉ mở khoá khi dùng `--finetune-fusion-lr` (xem bên dưới).
- `--limit 0.25`: biên độ gate residual. Không tự ý tăng biên độ chỉ để cải thiện training loss.
- `--adapt both` mặc định; dùng `--adapt base` hoặc `--adapt detail` cho ablation.
- `--finetune-fusion-lr LR` (mặc định 0, tắt): nếu > 0, mở khoá thêm `BaseFuseLayer`/`DetailFuseLayer` ở một nhóm tham số riêng với LR thấp hơn LR của adapter (ví dụ bằng 1/10 đến 1/100 `--lr`). Đây là chiến lược mở rộng không gian tham số học được (§4.1 trong [ANALYSIS.md](ANALYSIS.md)), không phải cách nới lỏng gate: mọi epoch vẫn phải qua đúng validation gate như khi chỉ học adapter.
- AdamW chỉ cập nhật các tham số đang mở khoá (adapter, và fusion layer nếu bật cờ trên), cosine learning rate, gradient clipping 1.0.
- Validation dùng full image, batch size 1, không augment.
- Seed được lưu; CUDA dùng deterministic khi có thể và cảnh báo với toán tử không hỗ trợ. Không bảo đảm bitwise reproducibility giữa các GPU/phiên bản thư viện.

Các output trong thư mục run:

| Output           | Ý nghĩa                                                                          |
| ---------------- | ---------------------------------------------------------------------------------- |
| `selected.pth` | Model được chọn; khởi đầu là original, chỉ thay khi gate validation đạt |
| `last.pth`     | Epoch cuối, có optimizer state; có thể kém original                           |
| `history.json` | Loss/metric mỗi epoch, cấu hình, hash, trạng thái lựa chọn                  |

`selected_epoch = -1` / `selection = baseline_retained` nghĩa là **chưa có cải tiến được chấp nhận**, không phải lỗi. Không tự chuyển sang last vì muốn thấy ảnh khác.

Gate: Qabf tăng hơn 0.0001; SSIM giảm không quá 0.001; SCD giảm không quá 0.01; clipping tăng không quá 0.001; proxy loss thấp hơn baseline. Trong các ứng viên đạt, lấy Qabf cao nhất. Chốt ngưỡng trước thí nghiệm, không chỉnh dựa vào test.

Nếu vẫn không có ứng viên đạt sau khi thử `--adapt both`/`base`/`detail` với objective hiện tại, thử thêm fine-tune fusion layer (mở rộng không gian tham số):

```powershell
python -m base_PFCFuse_improve.train_residual --cache base_PFCFuse_improve/cache/msrs_agf_wle --output base_PFCFuse_improve/runs/msrs_agf_wle_finetune_s3407 --epochs 20 --batch-size 4 --crop 128 --lr 0.001 --finetune-fusion-lr 0.00001 --seed 3407 --device cuda
```

Smoke test trên một vài ảnh thật, dùng output riêng:

```powershell
python -m base_PFCFuse_improve.data --source MSRS/train --output base_PFCFuse_improve/cache/smoke_agf_wle --mode agf --detail-fusion wle --limit 4
python -m base_PFCFuse_improve.train_residual --cache base_PFCFuse_improve/cache/smoke_agf_wle --output base_PFCFuse_improve/runs/smoke_agf_wle --epochs 1 --batch-size 1 --crop 32 --max-train-batches 1 --device cpu --threads 2
```

Ngay cả smoke này vẫn validation trên ảnh đầy đủ. `--limit` và `--max-train-batches` không dùng trong thí nghiệm báo cáo chất lượng.

## 7. Đánh giá test độc lập

Sau khi chọn cấu hình trên validation, tạo cache test riêng:

```powershell
python -m base_PFCFuse_improve.data --source MSRS/test --output base_PFCFuse_improve/cache/msrs_test_agf_wle --mode agf --detail-fusion wle --split test
python -m base_PFCFuse_improve.evaluation --cache base_PFCFuse_improve/cache/msrs_test_agf_wle --checkpoint base_PFCFuse_improve/runs/msrs_agf_wle_v3_s3407/selected.pth --output base_PFCFuse_improve/reports/msrs_test_wle_v3_s3407 --device cuda
```

Evaluator so sánh candidate và baseline cộng trên **cùng backbone frozen**, cùng cặp ảnh, AGF, WLE, clipping và lượng tử hóa 8-bit. Nó không gọi wrapper original đang import nhầm. Dataset mode, detail fusion và cấu hình WLE phải khớp checkpoint; cặp test có hash trùng train/validation sẽ bị từ chối.

Output gồm hai thư mục ảnh `original` / `improved`, `per_image.csv` và `report.json`. JSON chứa Qabf, SSIM trung bình hai nguồn, SCD, MI theo bit, entropy, spatial frequency, clipping, loss và bootstrap CI chênh lệch. `elapsed_seconds` gồm tính metric/xuất ảnh và chia sẻ encoder, **không phải benchmark tốc độ riêng hai mô hình**.

Chỉ coi `supported_on_this_split` là bằng chứng trên split đó. Script cần ít nhất 20 ảnh, gate đạt và cận dưới CI Qabf > 0. Nó không thay thế kiểm tra scene-level, ảnh trực quan hay tác vụ downstream. `not_established` nghĩa là chưa đủ bằng chứng, không nên tuyên bố chiến thắng.

Muốn xem lại validation: thêm `--split val` và dùng cache trainval; kết quả đó không phải test độc lập.

Đánh giá tổng quát hóa bằng lệnh tương tự với `--source data/TNO`, cache test riêng và giữ nguyên model/hyperparameter. Không tiếp tục chọn epoch dựa vào TNO nếu muốn gọi TNO là test.

## 8. Xuất ảnh dùng ngay

Sau khi có checkpoint được lựa chọn:

```powershell
python -m base_PFCFuse_improve.infer --source data/MSRS --checkpoint base_PFCFuse_improve/runs/msrs_agf_wle_v3_s3407/selected.pth --output data/output/update/pfcfuse_improve_wle --mode agf --detail-fusion wle --device cuda
```

Tạo baseline đúng kiến trúc gốc để đối chiếu:

```powershell
python -m base_PFCFuse_improve.infer --source data/MSRS --checkpoint base_PFCFuse_improve/PFCFuse_IVF.pth --output data/output/update/pfcfuse_original_wle --mode agf --detail-fusion wle --device cpu
```

`--limit 1` dùng kiểm tra nhanh một ảnh. Tên ảnh theo stem đầu vào, không hard-code 42/80 ảnh. Output phải chưa tồn tại; chọn tên khác khi chạy lại. `inference.json` lưu checkpoint hash, nguồn ảnh và cấu hình.

Để có đối chứng MLE, đổi hai tham số `--detail-fusion wle` thành `--detail-fusion mle`, dùng cache/checkpoint đã train bằng MLE và một thư mục output khác. Không dùng checkpoint WLE với `--detail-fusion mle`, hoặc ngược lại.

## 9. Tích hợp vào main hiện tại

[main.py](../main.py) đã chuyển detail fusion sang WLE:

```python
from base_PFCFuse_improve.base_PFCFuse import pfcfuse_fuse_gray
```

Và truyền checkpoint rõ ràng tại lời gọi:

```python
base_fused = pfcfuse_fuse_gray(
    base_ir,
    base_vi,
    ckpt_path="base_PFCFuse_improve/runs/msrs_agf_s3407/selected.pth",
)
```

Giữ nguyên AGF, `fuse_detail_dynamic_WLE_im2col_var_optimizer(detail_ir, detail_vi)` và `np.clip(base_fused + detail_fused, 0, 1)`. Đổi thư mục output để không ghi đè kết quả cũ. Thứ tự API là **IR trước, VIS sau**, khác thứ tự tensor của `FusionModel.forward(visible, infrared)`.

Nếu không truyền `ckpt_path`, wrapper cố ý dùng original trong package với adapter zero-init. Nó **không tự tìm model mới nhất**, không fallback im lặng khi đường dẫn bạn cung cấp bị sai. Đường dẫn default độc lập với working directory; đường dẫn tương đối do bạn truyền được giải theo working directory.

Đầu vào API là numpy 2D, finite, cùng shape, giá trị [0,1], mỗi chiều ít nhất 8. Đầu ra float32 [0,1], không min-max normalization. Dùng `baseline=True` để tắt adapter trong checkpoint improve; dùng `clear_model_cache()` khi cần giải phóng cache hoặc thay model trong cùng process.

## 10. Lỗi thường gặp

- `No train/val images`: cache được tạo với split sai hoặc thiếu manifest.
- `smaller than crop`: giảm `--crop`, không resize ảnh chỉ để vượt kiểm tra.
- `Train/validation overlaps` hoặc `Test overlaps`: sửa split nguồn, không bỏ guard.
- `strict state_dict` báo thiếu/thừa key: dùng checkpoint original đúng kiến trúc, không dùng checkpoint CMA/SCAF.
- `Checkpoint detail fusion does not match cache detail fusion`: tạo cache và train lại cùng `--detail-fusion`; không tái sử dụng cache/checkpoint MLE với WLE.
- CUDA không có: cài Torch phù hợp hoặc dùng `--device cpu`.
- `FileExistsError`: dùng thư mục output mới; chương trình không tự ghi đè.
- Warning `timm.models.layers` deprecated: đến từ net original được giữ nguyên; không ảnh hưởng các test đã chạy.

Để kết luận hiệu quả, tối thiểu chạy ablation base/detail/both với cùng split, ít nhất ba seed, test giữ kín và kiểm tra ảnh trực quan. Bản code không tự nhận rằng training loss giảm là model đã tốt hơn.
