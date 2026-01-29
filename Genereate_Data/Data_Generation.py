import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter, ImageChops
from google.colab import drive
from diffusers import StableDiffusionXLImg2ImgPipeline
from skimage.transform import swirl

# ==========================================
# 1. הגדרות המפעל
# ==========================================

# ==========================================
# 1. הגדרות המפעל
# ==========================================

BASE_DIR = "/content/drive/MyDrive/originals VS fake"
BRAND_NAME = "Converse"

# כמה לייצר
NUM_REAL_VARIATIONS = 10
NUM_REAL_LOW_QUAL = 2
NUM_AI_FAKES = 15
TYPO_PROB = 0.3

# תיקיות
SOURCE_DIR = os.path.join(BASE_DIR, f"original_logo/{BRAND_NAME}")

# מומלץ: לשמור real בתיקייה נפרדת כדי שלא יתערבב עם מקוריים
DEST_REAL = os.path.join(BASE_DIR, f"real_augmented/{BRAND_NAME}")

# זיופים
DEST_FAKE = os.path.join(BASE_DIR, f"counterfeit/{BRAND_NAME}")



# ==========================================
# 2. פונקציות עזר (איכות ועיוות)
# ==========================================

def make_image_low_quality(img):
    w, h = img.size
    img_bad = img.copy()
    img_bad = img_bad.resize((int(w/2.5), int(h/2.5)), resample=Image.BILINEAR)
    img_bad = img_bad.resize((w, h), resample=Image.BICUBIC)
    img_bad = img_bad.filter(ImageFilter.GaussianBlur(radius=0.8))
    img_np = np.array(img_bad)
    noise = np.random.normal(0, 12, img_np.shape).astype(np.uint8)
    img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)

def force_physical_distortion(img):
    img_np = np.array(img)
    distorted_np = swirl(img_np, rotation=0, strength=2, radius=400)
    distorted_np = (distorted_np * 255).astype(np.uint8)
    return Image.fromarray(distorted_np)

# ==========================================
# 2.1 זיופים נוספים (POST DEFECTS)
# ==========================================

def _mask_non_white(img, thr=30):
    """מסכה גסה: מה שלא לבן נחשב 'לוגו/תוכן'."""
    rgb = img.convert("RGB")
    arr = np.array(rgb).astype(np.int16)
    dist = np.abs(arr - 255).sum(axis=2)
    m = (dist > thr).astype(np.uint8) * 255
    mask = Image.fromarray(m, mode="L")
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1))
    return mask

def _apply_mask_as_alpha(img, mask):
    """מכניס mask כ-alpha ומחזיר קומפוזיט על רקע לבן."""
    rgba = img.convert("RGBA")
    rgba.putalpha(mask.convert("L"))
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba).convert("RGB")

def defect_thicken_logo(img):
    mask = _mask_non_white(img)
    k = random.choice([3, 5, 7])
    thick = mask.filter(ImageFilter.MaxFilter(k))
    return _apply_mask_as_alpha(img, thick)

def defect_thin_logo(img):
    mask = _mask_non_white(img)
    k = random.choice([3, 5, 7])
    thin = mask.filter(ImageFilter.MinFilter(k))
    return _apply_mask_as_alpha(img, thin)

def defect_partial_missing(img):
    mask = _mask_non_white(img)
    m = np.array(mask).copy()
    h, w = m.shape
    rw = random.randint(w // 10, w // 3)
    rh = random.randint(h // 10, h // 3)
    x0 = random.randint(0, max(0, w - rw))
    y0 = random.randint(0, max(0, h - rh))
    m[y0:y0+rh, x0:x0+rw] = 0
    return _apply_mask_as_alpha(img, Image.fromarray(m, "L"))

def defect_double_print(img):
    base = img.convert("RGB")
    layer = base.copy()
    dx = random.randint(-12, 12)
    dy = random.randint(-12, 12)
    shifted = Image.new("RGB", base.size, (255,255,255))
    shifted.paste(layer, (dx, dy))
    return Image.blend(base, shifted, alpha=0.35)

def defect_color_off(img):
    im = img.convert("RGB")
    im = ImageEnhance.Color(im).enhance(random.uniform(0.4, 1.7))
    im = ImageEnhance.Brightness(im).enhance(random.uniform(0.85, 1.15))
    im = ImageEnhance.Contrast(im).enhance(random.uniform(0.85, 1.15))
    return im

def defect_blurry_edges(img):
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 1.8)))

def defect_outline(img):
    base = img.convert("RGB")
    mask = _mask_non_white(base).convert("L")
    k = random.choice([5, 7, 9])
    dil = mask.filter(ImageFilter.MaxFilter(k))
    ring = ImageChops.subtract(dil, mask)
    ring = ring.filter(ImageFilter.GaussianBlur(radius=0.7))
    outline_layer = Image.new("RGB", base.size, (20, 20, 20))
    outlined = base.copy()
    outlined.paste(outline_layer, mask=ring)
    return outlined

def defect_edge_jitter(img):
    base = img.convert("RGB")
    mask = _mask_non_white(base).convert("L")
    m = np.array(mask).astype(np.int16)
    noise = np.random.normal(0, random.uniform(18, 35), m.shape).astype(np.int16)
    m2 = np.clip(m + noise, 0, 255).astype(np.uint8)
    thr = random.randint(110, 150)
    m2 = (m2 > thr).astype(np.uint8) * 255
    jitter_mask = Image.fromarray(m2, "L").filter(ImageFilter.GaussianBlur(radius=0.6))
    return _apply_mask_as_alpha(base, jitter_mask)

def defect_missing_letter(img):
    base = img.convert("RGB")
    mask = _mask_non_white(base).convert("L")
    m = np.array(mask).copy()
    h, w = m.shape
    rw = random.randint(max(6, w//25), max(10, w//10))
    rh = random.randint(h//3, int(h*0.9))
    x0 = random.randint(0, max(0, w - rw))
    y0 = random.randint(0, max(0, h - rh))
    m[y0:y0+rh, x0:x0+rw] = 0
    return _apply_mask_as_alpha(base, Image.fromarray(m, "L"))

def _find_perspective_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(pb).reshape(8)
    return np.linalg.lstsq(A, B, rcond=None)[0]

def defect_perspective_warp(img):
    base = img.convert("RGB")
    w, h = base.size
    margin = int(min(w, h) * random.uniform(0.03, 0.08))
    src = [(0,0), (w,0), (w,h), (0,h)]
    dst = [
        (random.randint(-margin, margin), random.randint(-margin, margin)),
        (w + random.randint(-margin, margin), random.randint(-margin, margin)),
        (w + random.randint(-margin, margin), h + random.randint(-margin, margin)),
        (random.randint(-margin, margin), h + random.randint(-margin, margin)),
    ]
    coeffs = _find_perspective_coeffs(dst, src)
    warped = base.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)
    if random.random() < 0.5:
        warped = warped.filter(ImageFilter.GaussianBlur(radius=0.6))
    return warped

POST_DEFECTS = [
    ("thin", defect_thin_logo),
    ("thicken", defect_thicken_logo),
    ("missing", defect_partial_missing),
    ("doubleprint", defect_double_print),
    ("coloroff", defect_color_off),
    ("blur", defect_blurry_edges),
    ("perspective", defect_perspective_warp),
    ("outline", defect_outline),
    ("edgejitter", defect_edge_jitter),
    ("missingletter", defect_missing_letter),
]

# ==========================================
# 2.2 Typo variants (כדי לשנות אותיות באמת)
# ==========================================

TYPO_VARIANTS = {
    "Nike": ["NYKE", "NIKY", "N1KE", "NIIE", "NKE","NKE","Hyke"],
    "Adidas": ["ABIBAS", "ADIDASS", "AD1DAS", "ADIDAS.","Adldas"],
    "Converse": ["CORNVERSE", "CONVERE", "CONVESE", "CONVERSE."],
    "Fila": ["FILLA", "HILA", "F1LA", "FILA.","FAIL","FIILA"],
    "New Balance": ["NEW BALARCE", "NEW BALENCE", "NEW BALANCE.", "NEWB ALANCE"],
    "Jordan": ["JORDON", "JORDAH", "J0RDAN", "JORDAN., JORDEN"]
}

# ==========================================
# 3. פונקציות הליבה
# ==========================================

def generate_authentic_variations(img, save_dir, base_name):
    generated_files = []
    os.makedirs(save_dir, exist_ok=True)

    for i in range(NUM_REAL_VARIATIONS):
        aug_img = img.copy()
        aug_img = aug_img.rotate(random.uniform(-5, 5), resample=Image.BICUBIC,
                                 expand=False, fillcolor=(255,255,255))
        aug_img = ImageEnhance.Brightness(aug_img).enhance(random.uniform(0.95, 1.05))
        save_path = os.path.join(save_dir, f"{base_name}_real_aug_{i}.jpg")
        aug_img.save(save_path)
        generated_files.append((save_path, "Real (Augmented)"))

    for i in range(NUM_REAL_LOW_QUAL):
        lq_img = make_image_low_quality(img)
        save_path = os.path.join(save_dir, f"{base_name}_real_bad_quality_{i}.jpg")
        lq_img.save(save_path)
        generated_files.append((save_path, "Real (Low Quality)"))

    return generated_files

def generate_ai_fakes(pipe, img, save_dir, base_name):
    generated_files = []
    os.makedirs(save_dir, exist_ok=True)

    prompts = {
        "Converse": [
            "A high-quality macro photo of a Converse circular patch, missing the star in the middle, canvas texture, detailed stitching",
            "Close-up of a vintage sneaker logo spelled 'Cornverse', realistic fabric weave, natural soft lighting, 8k photo",
            "An authentic looking shoe label with a typo 'Convese', detailed embroidery texture, realistic shadows",
            "Macro shot of a dirty worn-out Converse logo, star is distorted, realistic cotton material"
        ],
        "Adidas": [
            "Macro photo of an Adidas logo on fabric, spelled 'Abibas', 3D embroidery texture, realistic thread details",
            "Close up of a sports shoe label with 4 stripes, realistic leather texture, professional product photography",
            "High resolution shot of an Adidas heel tab, typo 'Adidass', leather grain texture, studio lighting"
        ],
        "Nike": [
            "A photorealistic macro shot of a leather sneaker heel, embroidered Nike logo spelled 'Niky', detailed leather grain, 8k resolution",
            "Close-up of a rubber Nike swoosh logo on a shoe, the swoosh is too thick and distorted, realistic rubber texture, soft lighting",
            "High quality product photo of a sneaker heel, typo 'Nik' missing the 'e', heavy stitching texture, sharp focus",
            "Macro photography of a Nike logo on fabric, spelled 'NIIE' with double I, realistic mesh texture, cinematic lighting"
        ],
        "Fila": [
            "Macro photo of a Fila sneaker tongue, embroidered logo spelled 'Filla' with double l, high detailed fabric texture",
            "Close-up of a Fila rubber logo patch, the letter 'F' is disconnected, realistic shoe material, studio light",
            "Photorealistic shot of a sneaker heel, typo 'Hila' instead of Fila, leather texture, intricate stitching details"
        ],
        "New Balance": [
            "High-quality macro photo of a New Balance 'N' logo on suede, missing the diagonal slashes, realistic suede texture",
            "Close-up of a sneaker heel tab, embroidered text 'New Balarce', detailed fabric weave, natural lighting",
            "A photorealistic shot of a New Balance logo, the 'N' is backwards, rich leather texture, 8k depth of field"
        ],
        "Jordan": [
            "Macro photo of a plastic Jordan heel tab, the Jumpman logo has a 'Fatman' silhouette, realistic glossy plastic texture, studio light",
            "Close-up of an embroidered Jumpman logo on leather, missing fingers on the hand, high detailed texture, 8k photo",
            "Photorealistic shot of a Jordan wing logo embossed on leather, spelled 'Jordon', sharp details, natural shadow",
            "High resolution close-up of a sneaker heel, fake Jordan logo with distorted legs, tumble leather texture"
        ]
    }

    brand_prompts = prompts.get(BRAND_NAME, ["close up photo of a fake brand logo on product, realistic texture"])
    img_resized = img.resize((768, 768))

    print(f" * מייצר {NUM_AI_FAKES} זיופים עבור {BRAND_NAME} (גם עדין וגם אגרסיבי)...")

    for i in range(NUM_AI_FAKES):
        base_prompt = random.choice(brand_prompts)

        # ---- שני מצבים: עדין (strength נמוך) או אגרסיבי (strength גבוה + typo) ----
        do_typo = (BRAND_NAME in TYPO_VARIANTS) and (random.random() < TYPO_PROB)
        typo_text = None

        if do_typo:
            typo_text = random.choice(TYPO_VARIANTS[BRAND_NAME])
            prompt = (
                f"{base_prompt}. The logo text is misspelled and clearly reads '{typo_text}'. "
                f"wrong spelling, incorrect typography, counterfeit print"
            )
            strength = random.uniform(0.70, 0.88)
            guidance = 10.0
            mode_name = "HIGH"
        else:
            prompt = base_prompt
            strength = random.uniform(0.45, 0.55)
            guidance = 8.0
            mode_name = "LOW"

        result = pipe(
            prompt=f"{prompt}, ultra photorealistic, 8k raw photo, highly detailed texture, sharp focus, cinematic lighting",
            negative_prompt="cartoon, drawing, illustration, anime, vector, 2d, digital art, flat painting, clip art, low quality, blurry",
            image=img_resized,
            strength=strength,
            guidance_scale=guidance
        )
        fake_img = result.images[0]

        # ---- מוסיפים דפקט רנדומלי כמו שיש לך עכשיו ----
        defect_name, defect_fn = random.choice(POST_DEFECTS)
        fake_img = defect_fn(fake_img)

        # ---- אופציונלי: פעם ב-10 תעשה גם "Distorted + Low Quality" חזק ----
        title_suffix = ""
        file_suffix = ""
        if (i % 10 == 9):  # כל עשירי
            fake_img = force_physical_distortion(fake_img)
            fake_img = make_image_low_quality(fake_img)
            title_suffix = "\n(Distorted)"
            file_suffix = "_distorted"

        typo_tag = f"_typo-{typo_text}" if typo_text else ""
        save_path = os.path.join(
            save_dir,
            f"{base_name}_fake_{i}_{mode_name}_s{strength:.2f}_{defect_name}{typo_tag}{file_suffix}.jpg"
        )
        fake_img.save(save_path)

        clean_prompt = base_prompt.split(',')[0]
        generated_files.append((save_path, f"Fake {i+1}{title_suffix}\n{mode_name} s={strength:.2f} | {defect_name}"))

    return generated_files

# ==========================================
# 4. הפונקציה הראשית
# ==========================================
def main():
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⏳ טוען מודל SDXL למכשיר {device}...")

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True
    ).to(device)

    if not os.path.isdir(SOURCE_DIR):
        print(f"* לא מצאתי תיקייה: {SOURCE_DIR}")
        return

    files = sorted([
        f for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ])

    if not files:
        print(f"* אין תמונות בתיקייה: {SOURCE_DIR}")
        return

    print(f"✅ נמצא {len(files)} תמונות. מריץ על כולן...")

    for idx, fname in enumerate(files, 1):
        file_path = os.path.join(SOURCE_DIR, fname)
        base_name = os.path.splitext(fname)[0]

        try:
            original_img = Image.open(file_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ דילוג על {fname} (לא נפתח): {e}")
            continue

        print(f"\n[{idx}/{len(files)}] ▶ עובד על: {fname}")

        # (אופציונלי) אם לא בא לך לייצר augment לכל קובץ – תעיף את השורה הזו
        generate_authentic_variations(original_img, DEST_REAL, base_name)

        generate_ai_fakes(pipe, original_img, DEST_FAKE, base_name)

    print("\n🎉 סיימתי! תבדוק את התיקייה:")
    print(f"   {DEST_FAKE}")


if __name__ == "__main__":
    main()
