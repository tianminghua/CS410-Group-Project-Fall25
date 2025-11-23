import json
import re

INPUT_FILE = "meta_Appliances.jsonl"
OUTPUT_FILE = "meta_Appliances_cleaned.jsonl"


def normalize_text(s):
    """简单文本清洗：lowercase + 去 URL/HTML/非字母数字 + 压缩空格"""
    if not s:
        return ""
    s = str(s)
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)        # 去 URL
    s = re.sub(r"<.*?>", " ", s)          # 去 HTML tag
    s = re.sub(r"[^a-z0-9\s']", " ", s)   # 非字母数字变空格
    s = re.sub(r"\s+", " ", s)            # 合并空格
    return s.strip()


def parse_price(price):
    """把 price 尽量转成 float，失败返回 None"""
    if price is None:
        return None
    if isinstance(price, (int, float)):
        return float(price)

    s = str(price)
    # 去掉 $、逗号 等，只保留数字和小数点
    s = re.sub(r"[^0-9.]", "", s)
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_brand(record):
    """优先用 details.Brand，其次用 store"""
    details = record.get("details") or {}
    brand = details.get("Brand")
    if not brand:
        brand = record.get("store")
    return brand


def flatten_categories(categories):
    """把 categories 展平成一维 list 并去重"""
    if not categories:
        return []
    flat = []
    for c in categories:
        if isinstance(c, list):
            flat.extend(c)
        else:
            flat.append(c)
    # 去重 + 去空
    flat_unique = [c for c in dict.fromkeys(flat) if c]
    return flat_unique


def build_all_text(record, brand, categories_str):
    """把用于搜索的文本拼成一个大字段 all_text"""
    parts = []

    title = record.get("title")
    if title:
        parts.append(title)

    features = record.get("features") or []
    description = record.get("description") or []

    parts.extend(features)
    parts.extend(description)

    if categories_str:
        parts.append(categories_str)
    if brand:
        parts.append(brand)
    if record.get("store"):
        parts.append(record["store"])

    raw_text = " ".join(str(p) for p in parts if p)
    return normalize_text(raw_text)


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
                # 坏行直接跳过
                continue

            parent_asin = record.get("parent_asin")
            title = record.get("title")

            # 没有 product id 或 title 的，意义不大，丢掉
            if not parent_asin or not title:
                continue

            main_category = record.get("main_category")

            # 类别处理
            categories_raw = record.get("categories") or []
            categories = flatten_categories(categories_raw)
            categories_str = " > ".join(categories) if categories else None

            # 品牌
            brand = extract_brand(record)

            # 评分相关
            avg_rating = record.get("average_rating")
            try:
                avg_rating = float(avg_rating) if avg_rating is not None else None
            except ValueError:
                avg_rating = None

            rating_number = record.get("rating_number")
            try:
                rating_number = int(rating_number) if rating_number is not None else None
            except ValueError:
                rating_number = None

            # 价格
            price = parse_price(record.get("price"))

            # 构造搜索文本
            all_text = build_all_text(record, brand, categories_str)

            cleaned = {
                "product_id": parent_asin,          # 用 parent_asin 作为统一 product id
                "title": title,
                "main_category": main_category,
                "categories": categories,
                "categories_str": categories_str,
                "brand": brand,
                "store": record.get("store"),
                "average_rating": avg_rating,
                "rating_number": rating_number,
                "price": price,
                "all_text": all_text
                # 注意：这里刻意不包括 images / videos / details 等复杂字段
            }

            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            num_out += 1

    print(f"Processed {num_in} lines, wrote {num_out} cleaned products to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
