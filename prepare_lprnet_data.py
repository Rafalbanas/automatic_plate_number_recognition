import os, random
import cv2
import xml.etree.ElementTree as ET

XML_PATH   = "/content/drive/MyDrive/zaawansowane_programowanie/automatic_plate_number_recognition/data/raw/poland-plates/annotations.xml"
IMAGES_DIR = "/content/drive/MyDrive/zaawansowane_programowanie/automatic_plate_number_recognition/data/raw/poland-plates/photos"
OUT_DIR    = "/content/drive/MyDrive/zaawansowane_programowanie/automatic_plate_number_recognition/lpr_data"
W, H       = 94, 24                          # docelowy rozmiar LPRNet (często działa najlepiej)
VAL_RATIO = 0.1
SEED = 42

os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)

def clamp(v, lo, hi): return max(lo, min(hi, v))

def crop_with_pad(img, x1,y1,x2,y2, pad=0.08):
    h, w = img.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad)
    py = int(bh * pad)
    x1 = clamp(x1 - px, 0, w-1)
    y1 = clamp(y1 - py, 0, h-1)
    x2 = clamp(x2 + px, 1, w)
    y2 = clamp(y2 + py, 1, h)
    return img[y1:y2, x1:x2]

# Uwaga: dopasuj pola do Twojego XML (czasem jest <object><bndbox>...</bndbox> + <name>PLATE_TEXT</name>)
tree = ET.parse(XML_PATH)
root = tree.getroot()

items = []
idx = 0

for img_node in root.findall(".//image"):
    # Jeśli masz format CVAT/LabelImg, trzeba dopasować. Poniżej “bezpieczny” wariant:
    file_name = img_node.get("name") or img_node.findtext("filename")
    if not file_name:
        continue
    img_path = os.path.join(IMAGES_DIR, file_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Przykład: obiekty/bboxy
    # 1) CVAT: <box xtl="" ytl="" xbr="" ybr="" label="plate" ...> + atrybut tekstu osobno
    # 2) PascalVOC: <object><name>ABC1234</name><bndbox>...</bndbox></object>
    # Tu próbujemy ogarnąć oba:
    voc_objs = img_node.findall(".//object")
    if voc_objs:
        for obj in voc_objs:
            label = (obj.findtext("name") or "").strip().upper()
            bb = obj.find("bndbox")
            if bb is None or len(label) < 4:
                continue
            xmin = bb.findtext("xmin")
            ymin = bb.findtext("ymin")
            xmax = bb.findtext("xmax")
            ymax = bb.findtext("ymax")
            if xmin is None or ymin is None or xmax is None or ymax is None:
                continue
            x1 = int(float(xmin))
            y1 = int(float(ymin))
            x2 = int(float(xmax))
            y2 = int(float(ymax))
            crop = crop_with_pad(img, x1,y1,x2,y2)
            crop = cv2.resize(crop, (W, H), interpolation=cv2.INTER_CUBIC)
            out_name = f"{idx:07d}.png"
            out_path = os.path.join(OUT_DIR, "images", out_name)
            cv2.imwrite(out_path, crop)
            items.append((out_path, label))
            idx += 1
    else:
        # CVAT-ish
        for box in img_node.findall(".//box"):
            xtl = box.get("xtl")
            ytl = box.get("ytl")
            xbr = box.get("xbr")
            ybr = box.get("ybr")
            if None in (xtl, ytl, xbr, ybr):
                continue
            x1 = int(float(str(xtl)))
            y1 = int(float(str(ytl)))
            x2 = int(float(str(xbr)))
            y2 = int(float(str(ybr)))
            # tekst bywa w atrybutach:
            label = (box.get("label") or "").strip().upper()
            # jeśli label to tylko "plate", a tekst jest w <attribute name="plate">ABC1234</attribute>
            for attr in box.findall(".//attribute"):
                if (attr.get("name") or "").lower() in ("plate", "text", "license", "lp"):
                    label = (attr.text or "").strip().upper()
            if len(label) < 4:
                continue
            crop = crop_with_pad(img, x1,y1,x2,y2)
            crop = cv2.resize(crop, (W, H), interpolation=cv2.INTER_CUBIC)
            out_name = f"{idx:07d}.png"
            out_path = os.path.join(OUT_DIR, "images", out_name)
            cv2.imwrite(out_path, crop)
            items.append((out_path, label))
            idx += 1

random.seed(SEED)
random.shuffle(items)

n = len(items)
nv = int(n * VAL_RATIO)
val = items[:nv]
train = items[nv:]

with open(os.path.join(OUT_DIR, "train.txt"), "w", encoding="utf-8") as f:
    for p, y in train:
        f.write(f"{p}\t{y}\n")

with open(os.path.join(OUT_DIR, "val.txt"), "w", encoding="utf-8") as f:
    for p, y in val:
        f.write(f"{p}\t{y}\n")

print("Saved:", n, "train:", len(train), "val:", len(val))
