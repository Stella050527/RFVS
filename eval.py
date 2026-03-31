import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import yaml
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ============================================================
# 1) 配置区：按你的实际路径改
# ============================================================
OUT_DIR = "/home/lhx/projects/20260128/runs/hard_replace_eval_4ways_mobilenet"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 8类检测/分类数据（用于 8cls / det+RFHard / 8cls+RFHard 的评测GT）
DATA_YAML_CLASS = "/home/lhx/projects/datasets/20260128-class.yaml"

# 1类检测数据（用于 det 的评测GT）
DATA_YAML_DET = "/home/lhx/projects/datasets/20260128-detect.yaml"

# 预测文件
PRED_8CLS_JSON = "/home/lhx/projects/20260128/runs/objClass/yolo11n_test/predictions.json"
PRED_DET_JSON  = "/home/lhx/projects/20260128/runs/objDetect/yolo11n_test/predictions.json"

# --- RF (MobileNetV3-Small) 分类器配置 ---
RF_TEST_DIR = "/home/lhx/AllData/All_split/RF_images/test"
RF_CKPT = "/home/lhx/AllData/All_split/RF_images/mobilenetv3_small_outputs/best_mobilenetv3_small.pth"
RF_MAPPING = "/home/lhx/AllData/All_split/RF_images/mobilenetv3_small_outputs/class_mapping.json"

# hard replace 规则
DELETE_IF_BACKGROUND = True
RF_MIN_PROB = 0.0                   # 0=纯hard replace；>0为门控
ONLY_REPLACE_TOP1_BOX = False       # True: 只替换最高分框；False: 整图替换
MULTIPLY_SCORE_BY_RF_PROB = False   # 可选：score *= rf_prob

# YOLO-style mp/mr 参数
YOLOSTYLE_IOU_THR = 0.5
# ============================================================


# ----------------------------
# utils
# ----------------------------
def norm_name(s: str) -> str:
    s = s.strip().lower()
    return "".join(ch for ch in s if ch.isalnum())


def load_data_yaml(data_yaml: str) -> Tuple[Path, List[str]]:
    with open(data_yaml, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    root = d.get("path", "")
    test_rel = d["test"]
    test_images_dir = Path(root) / test_rel
    names = d["names"]
    return test_images_dir, names


def list_images(images_dir: Path) -> List[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    imgs = []
    for e in exts:
        imgs.extend(images_dir.rglob(e))
    return sorted(set(imgs))


import math # 记得在文件头部导入 math

def yolo_to_coco_gt(images_dir: Path, names: List[str], out_json: str) -> str:
    imgs = list_images(images_dir)

    categories = [{"id": i + 1, "name": n} for i, n in enumerate(names)]  # 1-based
    images = []
    annotations = []
    ann_id = 1

    for img_id, img_path in enumerate(imgs, start=1):
        im = Image.open(img_path)
        W, H = im.size

        images.append({
            "id": img_id,
            "file_name": str(img_path),
            "width": W,
            "height": H,
        })

        label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        if not label_path.exists():
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls, xc, yc, w, h = line.split()
                cls = int(cls)  # 0-based
                xc, yc, w, h = map(float, (xc, yc, w, h))
                
                # --- 修改点 1：计算无量纲对角线长度 diag ---
                diag = math.sqrt(w**2 + h**2)

                bw = w * W
                bh = h * H
                x0 = (xc * W) - bw / 2
                y0 = (yc * H) - bh / 2

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls + 1,  # 1-based
                    "bbox": [x0, y0, bw, bh],
                    "area": diag,  # <-- 将原本的像素面积直接替换为 diag
                    "iscrowd": 0,
                })
                ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    return out_json

def build_stem_maps(coco_gt_json: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    coco_gt = COCO(coco_gt_json)
    stem_to_imgid: Dict[str, int] = {}
    imgid_to_file: Dict[int, str] = {}
    for img in coco_gt.dataset["images"]:
        img_id = int(img["id"])
        f = img["file_name"]
        imgid_to_file[img_id] = f
        stem_to_imgid[Path(f).stem] = img_id
    return stem_to_imgid, imgid_to_file


def normalize_pred_image_ids(preds: List[dict], stem_to_imgid: Dict[str, int]) -> List[dict]:
    out = []
    miss = 0
    for p in preds:
        img_id = None

        pred_file = p.get("file_name", None)
        if pred_file is not None:
            stem = Path(str(pred_file)).stem
            img_id = stem_to_imgid.get(stem, None)

        if img_id is None:
            pid = p.get("image_id", None)
            if pid is not None:
                stem = Path(str(pid)).stem
                img_id = stem_to_imgid.get(stem, None)

        if img_id is None:
            miss += 1
            continue

        p2 = dict(p)
        p2["image_id"] = int(img_id)
        out.append(p2)

    if miss > 0:
        print(f"[WARN] normalize_pred_image_ids: dropped {miss} preds (cannot map to GT image_id).")
    return out


# ----------------------------
# MobileNetV3 RF classifier
# ----------------------------
def build_mobilenetv3(num_classes: int, device: torch.device):
    """构建 MobileNetV3-Small 并修改分类头"""
    try:
        model = models.mobilenet_v3_small(weights=None)
    except TypeError:
        model = models.mobilenet_v3_small(pretrained=False)
    
    # MobileNetV3 的分类头是 classifier 序列，最后一层通常是 Linear
    # classifier 结构: Sequential(Linear, Hardswish, Dropout, Linear)
    # 我们修改最后一个 Linear 层 (index 3)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    model = model.to(device).eval()
    return model


def load_rf_model(device: torch.device):
    with open(RF_MAPPING, "r", encoding="utf-8") as f:
        m = json.load(f)
    classes = m["classes"]
    
    # 构建 MobileNetV3
    model = build_mobilenetv3(len(classes), device)

    ckpt = torch.load(RF_CKPT, map_location=device)
    # 兼容直接保存模型或保存 state_dict 的情况
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, classes, tf


@torch.no_grad()
def rf_predict(model, classes: List[str], tf, img_path: Path, device: torch.device) -> Tuple[str, float]:
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    idx = int(torch.argmax(probs).item())
    return classes[idx], float(probs[idx].item())


def find_rf_image_for_vision_stem(stem: str) -> Optional[Path]:
    cands = list(Path(RF_TEST_DIR).rglob(f"*{stem}*.png"))
    if cands:
        return cands[0]
    short = stem[:20]
    cands = list(Path(RF_TEST_DIR).rglob(f"*{short}*.png"))
    if cands:
        return cands[0]
    return None


# ----------------------------
# Replacement Logic
# ----------------------------
def rf_replace_to_target_classes(
    preds_normed: List[dict],
    coco_gt_json: str,
    target_names: List[str],
    out_json: str,
    mode: str = "hard",          # "hard" or "gate"
    gate_thr: float = 0.8,       # gate阈值
    gate_bg: bool = True,        # background 也参与gate；True: p>=thr才删框；False: 一旦bg就删
) -> List[dict]:
    """
    用 RF (MobileNet) 预测的整图类别替换该图所有框的 category_id（框不变）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, rf_classes, tf = load_rf_model(device)

    target_norm_to_cid = {norm_name(n): i + 1 for i, n in enumerate(target_names)}  # 1-based

    _, imgid_to_file = build_stem_maps(coco_gt_json)

    by_img: Dict[int, List[dict]] = defaultdict(list)
    for p in preds_normed:
        by_img[int(p["image_id"])].append(p)

    new_preds: List[dict] = []

    for img_id, dets in by_img.items():
        gt_file = imgid_to_file.get(int(img_id), "")
        vision_stem = Path(gt_file).stem

        rf_img = find_rf_image_for_vision_stem(vision_stem)
        if rf_img is None:
            # 找不到RF：保持原结果
            new_preds.extend(dets)
            continue

        rf_name, rf_prob = rf_predict(model, rf_classes, tf, rf_img, device)
        rf_norm = norm_name(rf_name)

        # 先处理 background
        if rf_norm == norm_name("background") and DELETE_IF_BACKGROUND:
            if mode == "hard":
                continue
            elif mode == "gate":
                if gate_bg:
                    if rf_prob >= gate_thr:
                        continue
                    else:
                        new_preds.extend(dets)   # 不够自信，不删
                        continue
                else:
                    continue

        # 置信度门控（全局最低置信度）
        if rf_prob < RF_MIN_PROB:
            new_preds.extend(dets)
            continue

        # gate 模式：不够阈值就不替换
        if mode == "gate" and rf_prob < gate_thr:
            new_preds.extend(dets)
            continue

        # RF 类名映射到目标类别集合
        if rf_norm not in target_norm_to_cid:
            new_preds.extend(dets)
            continue

        new_cid = int(target_norm_to_cid[rf_norm])

        if ONLY_REPLACE_TOP1_BOX:
            dets_sorted = sorted(dets, key=lambda x: float(x.get("score", 0.0)), reverse=True)
            for j, p in enumerate(dets_sorted):
                p2 = dict(p)
                if j == 0:
                    p2["category_id"] = new_cid
                    if MULTIPLY_SCORE_BY_RF_PROB:
                        p2["score"] = float(p2["score"]) * float(rf_prob)
                new_preds.append(p2)
        else:
            for p in dets:
                p2 = dict(p)
                p2["category_id"] = new_cid
                if MULTIPLY_SCORE_BY_RF_PROB:
                    p2["score"] = float(p2["score"]) * float(rf_prob)
                new_preds.append(p2)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(new_preds, f)
    return new_preds


def hard_replace_to_target_classes(
    preds_normed: List[dict],
    coco_gt_json: str,
    target_names: List[str],
    out_json: str,
) -> List[dict]:
    """
    用 RF 预测的整图类别替换该图所有框的 category_id（框不变）。
    这里是纯 Hard 模式的封装，内部直接调用通用逻辑，参数固定为 mode='hard'
    """
    return rf_replace_to_target_classes(
        preds_normed=preds_normed,
        coco_gt_json=coco_gt_json,
        target_names=target_names,
        out_json=out_json,
        mode="hard"
    )


# ----------------------------
# Metrics
# ----------------------------
def coco_eval_metrics(gt_json: str, pred_json: str) -> Dict[str, float]:
    import math
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)
    
    # --- 修改点 2.1：修正预测结果 (DT) 中的 area 为 diag ---
    # 获取图片的宽高，用于将 DT 的绝对坐标转换回顾一化宽高
    img_id_to_dim = {img['id']: (img['width'], img['height']) for img in coco_gt.dataset['images']}
    
    for ann in coco_dt.dataset['annotations']:
        # ann['bbox'] 是绝对坐标 [x, y, bw, bh]
        bw, bh = ann['bbox'][2], ann['bbox'][3]
        W, H = img_id_to_dim[ann['image_id']]
        
        # 反推归一化宽和高
        w_norm, h_norm = bw / W, bh / H
        diag = math.sqrt(w_norm**2 + h_norm**2)
        ann['area'] = diag  # 用 diag 覆盖原有的绝对像素面积
        
    # 刷新内部索引确保修改生效
    coco_dt.createIndex()

    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    
    # --- 修改点 2.2：忽略纯背景图片 ---
    # 提取所有包含真实 GT 边界框的 image_id
    non_bg_img_ids = list(set([ann['image_id'] for ann in coco_gt.dataset['annotations']]))
    ev.params.imgIds = non_bg_img_ids
    
    # --- 修改点 2.3：重写评估区间 (Area Ranges) ---
    # 替换标准 COCO 像素阈值，改为论文中基于 diag 的阈值
    # Inst_s: [0, 0.02), Inst_m: [0.02, 0.04), Inst_l: [0.04, 2.0 (使用 2.0 作为足够大的上界)]
    ev.params.areaRng = [[0, 100000.0], [0, 0.02], [0.02, 0.04], [0.04, 2.0]]
    ev.params.areaRngLbl = ['all', 'small', 'medium', 'large']

    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    
    # s[3], s[4], s[5] 对应于重写区间后的 mAP50-95 (Small, Medium, Large)
    return {
        "mAP": float(s[0]),
        "mAP50": float(s[1]),
        "mAP75": float(s[2]),
        "mAP_s": float(s[3]),
        "mAP_m": float(s[4]),
        "mAP_l": float(s[5]),
    }

def bbox_iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_x1, a_y1, a_x2, a_y2 = ax, ay, ax + aw, ay + ah
    b_x1, b_y1, b_x2, b_y2 = bx, by, bx + bw, by + bh
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union


def yolo_style_mp_mr_bestf1(gt_json: str, pred_json: str, iou_thr: float = 0.5) -> Dict[str, float]:
    coco = COCO(gt_json)
    preds = json.load(open(pred_json, "r", encoding="utf-8"))

    cat_ids = sorted([c["id"] for c in coco.dataset["categories"]])
    img_ids = sorted([img["id"] for img in coco.dataset["images"]])

    gt_instances = len(coco.dataset["annotations"])
    num_images = len(img_ids)

    gt_by_img_cat = defaultdict(lambda: defaultdict(list))
    for ann in coco.dataset["annotations"]:
        gt_by_img_cat[int(ann["image_id"])][int(ann["category_id"])].append(ann["bbox"])

    pred_by_cat = defaultdict(list)
    for p in preds:
        cid = int(p["category_id"])
        pred_by_cat[cid].append((int(p["image_id"]), float(p["score"]), p["bbox"]))

    per_class_P = []
    per_class_R = []

    for cid in cat_ids:
        dets = pred_by_cat.get(cid, [])
        dets.sort(key=lambda x: x[1], reverse=True)

        n_gt = sum(len(gt_by_img_cat[img_id].get(cid, [])) for img_id in img_ids)
        if n_gt == 0:
            continue

        matched = {img_id: np.zeros(len(gt_by_img_cat[img_id].get(cid, [])), dtype=bool) for img_id in img_ids}

        tp = np.zeros(len(dets), dtype=np.float32)
        fp = np.zeros(len(dets), dtype=np.float32)

        for i, (img_id, score, bb) in enumerate(dets):
            gts = gt_by_img_cat[img_id].get(cid, [])
            if len(gts) == 0:
                fp[i] = 1
                continue

            ious = [bbox_iou_xywh(bb, gtbb) for gtbb in gts]
            j = int(np.argmax(ious))
            best_iou = ious[j]

            if best_iou >= iou_thr and not matched[img_id][j]:
                tp[i] = 1
                matched[img_id][j] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recall = tp_cum / (n_gt + 1e-12)
        precision = tp_cum / (tp_cum + fp_cum + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        k = int(np.argmax(f1))
        per_class_P.append(float(precision[k]))
        per_class_R.append(float(recall[k]))

    mp = float(np.mean(per_class_P)) if per_class_P else 0.0
    mr = float(np.mean(per_class_R)) if per_class_R else 0.0

    return {"Images": int(num_images), "Instances": int(gt_instances), "mp": mp, "mr": mr}


def save_and_eval(tag: str, gt_json: str, pred_list: List[dict], save_path: str) -> Dict[str, float]:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(pred_list, f)

    coco_m = coco_eval_metrics(gt_json, save_path)
    yolo_m = yolo_style_mp_mr_bestf1(gt_json, save_path, iou_thr=YOLOSTYLE_IOU_THR)

    return {
        "Model": tag,
        "Images": yolo_m["Images"],
        "Instances": yolo_m["Instances"],
        "mp": yolo_m["mp"],
        "mr": yolo_m["mr"],
        **coco_m,
    }


def print_table(title: str, rows: List[Dict[str, float]]):
    print("\n====================", title, "====================")
    print("Model\tImages\tInstances\tmp\tmr\tmAP50\tmAP50-95\tmAP75\tmAP_s\tmAP_m\tmAP_l")
    for r in rows:
        print(
            f"{r['Model']}\t{r['Images']}\t{r['Instances']}\t"
            f"{r['mp']:.3f}\t{r['mr']:.3f}\t"
            f"{r['mAP50']:.3f}\t{r['mAP']:.3f}\t{r['mAP75']:.3f}\t"
            f"{r['mAP_s']:.3f}\t{r['mAP_m']:.3f}\t{r['mAP_l']:.3f}"
        )


if __name__ == "__main__":
    # --------- Build GTs ---------
    test_dir_class, names_class = load_data_yaml(DATA_YAML_CLASS)
    test_dir_det, names_det = load_data_yaml(DATA_YAML_DET)

    gt_class = str(Path(OUT_DIR) / "instances_test_class.json")
    gt_det = str(Path(OUT_DIR) / "instances_test_det.json")

    if not Path(gt_class).exists():
        print("[GT] building class GT...")
        yolo_to_coco_gt(test_dir_class, names_class, gt_class)
    if not Path(gt_det).exists():
        print("[GT] building det GT...")
        yolo_to_coco_gt(test_dir_det, names_det, gt_det)

    stem2id_class, _ = build_stem_maps(gt_class)
    stem2id_det, _ = build_stem_maps(gt_det)

    # --------- Load predictions ---------
    preds_8cls_raw = json.load(open(PRED_8CLS_JSON, "r", encoding="utf-8"))
    preds_det_raw  = json.load(open(PRED_DET_JSON, "r", encoding="utf-8"))


    # 8cls 在 class GT 上归一化
    preds_8cls_on_class = normalize_pred_image_ids(preds_8cls_raw, stem2id_class)

    # det 在 det GT 上归一化（用于 1类检测评测）
    preds_det_on_det = normalize_pred_image_ids(preds_det_raw, stem2id_det)

    # det 在 class GT 上归一化（用于 det+RFHard 的 8类评测）
    preds_det_on_class = normalize_pred_image_ids(preds_det_raw, stem2id_class)

    # --------- 1) det (1-class) ---------
    det_row = save_and_eval(
        "det",
        gt_det,
        preds_det_on_det,
        str(Path(OUT_DIR) / "pred_det_baseline.json"),
    )

    # --------- 2) 8cls baseline (8-class) ---------
    cls_row = save_and_eval(
        "8cls",
        gt_class,
        preds_8cls_on_class,
        str(Path(OUT_DIR) / "pred_8cls_baseline.json"),
    )

    # --------- 3) det + RFHard (8-class eval on class GT) ---------
    pred_det_rf = hard_replace_to_target_classes(
        preds_det_on_class,
        gt_class,
        names_class,  # 输出必须映射到8类names
        str(Path(OUT_DIR) / "pred_det_RFHard_list.json"),
    )
    det_rf_row = save_and_eval(
        "det+RFHard",
        gt_class,
        pred_det_rf,
        str(Path(OUT_DIR) / "pred_det_RFHard.json"),
    )

    # --------- 4) 8cls + RFHard (8-class) ---------
    pred_8cls_rf = hard_replace_to_target_classes(
        preds_8cls_on_class,
        gt_class,
        names_class,
        str(Path(OUT_DIR) / "pred_8cls_RFHard_list.json"),
    )
    cls_rf_row = save_and_eval(
        "8cls+RFHard",
        gt_class,
        pred_8cls_rf,
        str(Path(OUT_DIR) / "pred_8cls_RFHard.json"),
    )
    # --------- 5) 8cls + RFGate (8-class) ---------
    pred_8cls_gate = rf_replace_to_target_classes(
        preds_normed=preds_8cls_on_class,
        coco_gt_json=gt_class,
        target_names=names_class,
        out_json=str(Path(OUT_DIR) / "pred_8cls_RFGate_list.json"),
        mode="gate",
        gate_thr=0.8,      # 门控阈值
        gate_bg=True,      # background 也用同阈值才删
    )
    cls_gate_row = save_and_eval(
        "8cls+RFGate",
        gt_class,
        pred_8cls_gate,
        str(Path(OUT_DIR) / "pred_8cls_RFGate.json"),
    )


    # --------- Print tables ---------
    print_table("1-class DET evaluation (detect.yaml GT)", [det_row])
    print_table("8-class evaluation (class.yaml GT)", [cls_row, det_rf_row, cls_rf_row, cls_gate_row]) 改成英文注释，然后敏感的路径删掉并提示怎么写路径
