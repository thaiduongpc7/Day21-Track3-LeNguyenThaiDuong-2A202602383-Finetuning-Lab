# Lab 21 - Evaluation Report

**Ngay chay:** 2026-10-07  
**Trang thai nop:** da chay NB1-NB5 tren local Windows + NVIDIA GeForce RTX 4060 Laptop GPU 8GB.  
**Phan quyet:** FAIL, nhung la FAIL hop le va co bang chung: fine-tune thang manh o target task, nhung lam hong regression/general capability qua nguong cho phep.

## 1. Model, dataset va ly do chon

Toi chon `Qwen/Qwen3.5-0.8B` voi tier `LAPTOP`. Ly do thuc dung la may local co RTX 4060 Laptop 8GB; cac model lon hon nhu mac dinh T4/4B va bien the 2B kho tai/chay on dinh trong moi truong nay. Viec doi model van giu dung nguyen tac cong bang cua lab: NB2 baseline va NB3/NB5 fine-tune dung cung mot base model, cung tap eval, cung optimized prompt da dong bang.

Dataset la dataset mac dinh cua repo: 250 ticket cham soc khach hang tieng Viet, output JSON triage gom 4 truong `intent`, `urgency`, `product`, `sentiment`. Toi khong doi noi dung tap eval. Tren Windows, line ending ban dau co the thanh CRLF lam checksum bytes bi lech; toi normalize cac file data ve LF va them `.gitattributes` de checksum cua gatekeeper phan anh dung noi dung goc.

Thong so chinh:

| Muc | Gia tri |
|---|---|
| Tier | `LAPTOP` |
| Base model | `Qwen/Qwen3.5-0.8B` |
| Dataset | default CSKH ticket -> JSON triage |
| Train/val | 225/25, split seed 42 |
| `MASK_MODE` | `assistant-only` |
| `EPOCHS` | 2 |
| `MAX_LENGTH` | 256 |
| Full eval | co, `EVAL_LIMIT=null`, `smoke_mode=false` |

`MAX_LENGTH=256` khong doan tay. File `results/token_stats.json` do tren 250 mau cho `p95=98`, `p99=100`, `max=101`, `suggested_max_length=256`.

## 2. Bang chung mask - NB1

NB1 da tao du 3 file: `results/template_check.json`, `results/mask_proof.json`, `results/token_stats.json`.

`template_check.json` cho thay chat template giu khoi suy nghi:

| Muc | Gia tri |
|---|---|
| `ok` | `true` |
| `open_tag_present` | `true` |
| `body_present` | `true` |
| Verdict | reasoning preserved - safe to train on traces |

`mask_proof.json` la bang chung quan trong nhat:

| Muc | Gia tri |
|---|---:|
| `mask_mode` | `assistant-only` |
| `n_supervised` | 37 |
| `n_total` | 94 |
| `supervised_fraction` | 0.3936 |
| `answer_is_supervised` | `true` |
| `question_is_masked` | `true` |

Doan co loss la dung phan tra loi JSON:

```text
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Prompt, user ticket va scaffold `<think></think>` nam ngoai loss. `supervised_fraction=0.3936`, thap hon xa nguong mat diem `0.95`, nen run nay khong train tren ca prompt.

Sau khi doi base model, toi chay `python scripts/check_mask_agreement.py`. Ket qua can dien giai: template cua model khong co `{% generation %}`, nen duong tokenizer-level cua TRL tao assistant mask rong va chi canh bao. Vi vay NB3 dung pre-tokenized `labels` tu NB1, khong dua vao co `assistant_only_loss`.

## 3. Baseline dong bang - NB2

NB2 da chay truoc train va ghi `results/baselines_frozen.json`. Optimized prompt SHA la `719e74d3b6232053`. Tap eval dung day du: `n_target=50`, `n_regression=15`, `eval_limit=null`.

| Run | target | regression | format | latency ms | n |
|---|---:|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.6444 | 0.000 | 1779.9 | 50 |
| (b) base + optimized prompt | 0.495 | 0.6444 | 1.000 | 482.4 | 50 |

Dieu kien NB2 `(b) > (a)` dat ro rang: target tu `0.000` len `0.495`. Day la doi thu that cua fine-tune, khong phai prompt yeu.

## 4. Train dung va ba cau hinh doi chung - NB3/NB4

NB3 sinh `adapters/correct/` va ghi dong `correct` trong `results/runs.csv`. Cau hinh dung: all-linear LoRA, `r=16`, `alpha=32`, `learning_rate=1e-4`, bf16, train tren pre-tokenized mask da chung minh. Model in `layer_types`: 24 hidden layers, 18 `linear_attention`, 6 `full_attention`, `full_attention_interval=4`, `linear_num_key_heads=16`.

NB4 da chay duoc hai doi chung `attn_only` va `wrong_lr` cung `max_steps=58`. `qlora` khong chay tren local Windows vi moi truong khong co `bitsandbytes` hoat dong; day la gioi han moi truong, khong phai ket qua duoc tinh la mot run thanh cong.

| Run | Vi tri / loi co y | r | alpha | LR | trainable params | final loss | VRAM GB | steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | all-linear, LR 10x | 16 | 32 | 0.0001 | 10,822,656 | 0.3912 | 1.97 | 58 |
| `attn_only` | chi q,v, rank matched | 271 | 542 | 0.0001 | 10,822,656 | 0.4329 | 1.98 | 58 |
| `wrong_lr` | all-linear, LR full-FT scale | 16 | 32 | 0.00001 | 10,822,656 | 1.5439 | 1.98 | 58 |

`attn_only` la so sanh cong bang ve ngan sach tham so: 10,822,656 vs 10,822,656 trainable params, sai lech 0%, nho hon nguong 5%. Neu chi so sanh q,v rank 16 voi all-linear rank 16 thi do la so sanh ngan sach, khong phai so sanh vi tri; run nay tranh loi do bang `matched_rank()`.

## 5. Ket qua NB5 va phan quyet

`results/verdict.json` ghi phan quyet FAILED:

| Run | target | regression | format | latency ms | n |
|---|---:|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.6444 | 0.000 | 1779.9 | 50 |
| (b) base + optimized prompt | 0.495 | 0.6444 | 1.000 | 482.4 | 50 |
| (c) LoRA fine-tune | 0.990 | 0.0667 | 1.000 | 648.8 | 50 |

Fine-tune thang baseline prompt o target: `target_delta=+0.495`. Tuy nhien regression tut tu `0.6444` xuong `0.0667`, tuc `regression_delta=-0.5778`, vuot xa tolerance `0.020`. Vi vay verdict FAIL la dung. Ket luan cua toi: adapter hoc task JSON triage rat tot nhung bi over-specialize vao format/tap CSKH, lam mat nang luc tra loi cau hoi pho thong. Run nay khong nen trien khai neu chua them replay data hoac mix instruction data de bao ve general capability.

Ket qua xep hang cac run bang target NB5, khong bang final loss NB4:

| Run | target | format | latency ms | n |
|---|---:|---:|---:|---:|
| `correct` | 0.990 | 1.000 | 648.8 | 50 |
| `attn_only` | 0.925 | 1.000 | 457.5 | 50 |
| `wrong_lr` | 0.335 | 0.995 | 635.9 | 50 |

Nhan xet: `correct` tot nhat tren target, `attn_only` cung kha cao khi da duoc match tham so nhung van thua all-linear, con `wrong_lr` thua nang. Dieu nay ung voi bai hoc trong deck: LoRA can LR lon hon full fine-tune scale; final loss cua train run chi la tin hieu phu, diem target NB5 moi la tieu chi xep hang.

## 6. Vi du dinh tinh

`results/qualitative.json` luu cac ca target theo score cua fine-tune. Hai ca te nhat deu dat `0.75`, chu yeu vi output bi cat ngan trong preview/prediction:

| Nhom | i | Nhan xet |
|---|---:|---|
| target bi loi | 18 | Ticket ve may xay sinh to, intent hoan_tien. Fine-tune bat dung intent/product nhung preview JSON bi cat o `sentiment`, score 0.75. |
| target bi loi | 23 | Ticket ve noi chien khong dau, giao hang cham. Fine-tune bat dung intent van_chuyen/product nhung output bi cat truoc khi hoan tat day du, score 0.75. |
| target tot | 47 | Ticket ve op lung dien thoai va shipper; fine-tune tra JSON dung, score 1.00. |
| target tot | 48 | Ticket hoi gia op lung dien thoai; fine-tune tra `hoi_thong_tin`, score 1.00. |
| target tot | 49 | Ticket sai mau op lung dien thoai; fine-tune tra `san_pham_loi`, score 1.00. |

Hai ca fine-tune thua quan trong nam o nhom regression. `results/verdict.json` khong luu tung prediction regression, nhung diem tong the da du ro: base optimized prompt dat `0.6444`, fine-tune chi dat `0.0667`. Cac cau regression gom nhung cau pho thong nhu "Thu do cua Viet Nam la thanh pho nao?" va "1 km bang bao nhieu met?". Adapter sau fine-tune khong con giu du nang luc nay, nen day la bang chung thua that su so voi baseline prompt (b), du target task tang manh.

## 7. Dieu hoc duoc

1. Mask phai duoc chung minh bang cach decode nguoc `labels != -100`. Neu chi tin vao trainer flag, co the gap dung truong hop template thieu generation marker va mask rong.

2. Prompt baseline manh la dieu kien cong bang. Trong run nay optimized prompt day base model tu target `0.000` len `0.495`; fine-tune chi co y nghia khi thang moc nay, khong phai moc prompt yeu.

3. Fine-tune co the "thang task" nhung van FAIL. `target=0.990` rat dep, nhung `regression=0.0667` lam adapter khong dat tieu chuan. Ket qua nay thuyet phuc hon mot report chi noi "fine-tune tot" vi no chi ra chi phi cua specialization.

4. Doi chung can cong bang ve ngan sach. `attn_only` phai duoc tang rank len 271 de khop tham so voi all-linear rank 16; neu khong, ket luan ve placement se bi nhiem boi boi parameter budget.

5. Tren may Windows, QLoRA khong phai luc nao cung kha thi vi `bitsandbytes`. Neu can nop day du NB4 core, cach tot hon la chay lai NB4 QLoRA tren Colab/Linux. Trong bo ket qua local nay, qlora la han che da khai bao, khong duoc dien thanh so lieu gia.
