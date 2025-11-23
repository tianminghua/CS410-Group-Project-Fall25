import json
import re

# === 配置 ===
INPUT_FILE = "Appliances.jsonl"
OUTPUT_FILE = "Appliances_cleaned.jsonl"


def normalize_text(s):
    """简单文本清洗：lowercase + 去 URL/HTML/非字母数字 + 压缩空格"""
    if not s:
        return ""
    s = str(s)
    s = s.lower()
    # 去 URL
    s = re.sub(r"http\S+", " ", s)
    # 去 HTML tag
    s = re.sub(r"<.*?>", " ", s)
    # 只保留字母、数字、空格和 ' （方便 don't）
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    # 合并空格
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def safe_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x))
    except ValueError:
        return None


def safe_int(x):
    if x is None:
        return None
    if isinstance(x, int):
        return x
    try:
        return int(str(x))
    except ValueError:
        return None


def parse_verified_purchase(x):
    """
    可靠解析 verified_purchase：
    - True / False
    - "true"/"false"
    - 1 / 0
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y"):
            return True
        if s in ("false", "0", "no", "n"):
            return False
    return None


def main():
    num_in = 0
    num_out = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            num_in += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = record.get("asin")
            parent_asin = record.get("parent_asin")
            title = record.get("title") or ""
            text = record.get("text") or ""

            # 没 asin 或 title+text 都空 → 跳过
            if not asin or (not title.strip() and not text.strip()):
                continue

            # 过滤：只保留 verified_purchase == True
            verified = parse_verified_purchase(record.get("verified_purchase"))
            if verified is not True:
                continue

            # 生成 clean_content
            merged = (title + " " + text).strip()
            clean_content = normalize_text(merged)

            # 太短的无意义评论删掉
            if len(clean_content) < 10:
                continue

            cleaned = {
                "asin": asin,
                "parent_asin": parent_asin,
                "rating": safe_float(record.get("rating")),
                "title": title,
                "text": text,
                "user_id": record.get("user_id"),
                "timestamp": safe_int(record.get("timestamp")),
                "helpful_vote": safe_int(record.get("helpful_vote")),
                # 不再保存 verified_purchase 字段（节省空间）
                "clean_content": clean_content
            }

            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            num_out += 1

    print(f"Processed {num_in} lines, wrote {num_out} filtered verified reviews to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
