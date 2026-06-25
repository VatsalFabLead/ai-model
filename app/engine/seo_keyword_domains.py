"""Master domain classifier — 210 SEO domains across 21 categories (IDs 1–210)."""

from __future__ import annotations

import re
from typing import Any

# Category → ordered domain list (stable IDs assigned in catalog order)
DOMAIN_CATALOG: dict[str, list[str]] = {
  "Technology": [
    "Technology", "Software", "SaaS", "Artificial Intelligence", "Machine Learning",
    "Deep Learning", "Generative AI", "Data Science", "Big Data", "Cloud Computing",
    "Cybersecurity", "Blockchain", "Cryptocurrency", "Web3", "Mobile App Development",
    "Web Development", "UI/UX Design", "DevOps", "Networking", "IoT",
  ],
  "Healthcare & Medical": [
    "Healthcare", "Hospital", "Clinic", "Physiotherapy", "Dentist", "Pharmacy",
    "Laboratory", "Mental Health", "Veterinary", "Medical Equipment", "Telemedicine",
    "Diagnostics", "Nutrition", "Fitness", "Yoga", "Wellness",
  ],
  "Beauty & Fashion": [
    "Beauty", "Cosmetics", "Skincare", "Haircare", "Makeup", "Perfume", "Fashion",
    "Clothing", "Jewelry", "Watches", "Footwear", "Accessories",
  ],
  "Food": [
    "Restaurant", "Cafe", "Bakery", "Fast Food", "Food Delivery", "Grocery",
    "Catering", "Beverage", "Organic Food", "Nutrition",
  ],
  "Automotive": [
    "Automobile", "Luxury Cars", "Used Cars", "Bikes", "EV Vehicles", "Car Rental",
    "Taxi", "Auto Parts", "Car Service", "Car Accessories",
  ],
  "Travel": [
    "Travel", "Tourism", "Hotels", "Resorts", "Flights", "Cruise", "Visa",
    "Adventure", "Trekking", "Camping", "Holiday Packages", "Travel Insurance",
  ],
  "Real Estate": [
    "Real Estate", "Property", "Apartments", "Villas", "Commercial Property",
    "Interior Design", "Architecture", "Construction", "Home Decor", "Furniture",
  ],
  "Finance": [
    "Finance", "Banking", "Insurance", "Investment", "Stock Market", "Mutual Funds",
    "Loans", "Credit Cards", "Tax", "Accounting",
  ],
  "Education": [
    "Education", "School", "College", "University", "Online Learning", "Coaching",
    "Certification", "EdTech", "Competitive Exams", "Language Learning",
  ],
  "Business": [
    "Business", "Startup", "Consulting", "Marketing", "Digital Marketing", "SEO",
    "Content Marketing", "Email Marketing", "Branding", "Advertising",
  ],
  "Ecommerce": [
    "Ecommerce", "Marketplace", "Dropshipping", "Wholesale", "Retail", "B2B", "B2C",
    "Logistics", "Warehousing", "Shipping",
  ],
  "Entertainment": [
    "Movies", "TV Shows", "Music", "OTT", "Gaming", "Esports", "Streaming",
    "Celebrity", "Events", "Photography",
  ],
  "Sports": [
    "Sports", "Cricket", "Football", "Basketball", "Tennis", "Badminton", "Gym",
    "Bodybuilding", "Running", "Cycling",
  ],
  "Government & Legal": [
    "Government", "Legal", "Law Firm", "Court", "Immigration", "Passport",
    "Public Services", "NGO", "Charity", "Public Policy",
  ],
  "Industrial": [
    "Manufacturing", "Machinery", "Chemicals", "Mining", "Agriculture", "Dairy",
    "Renewable Energy", "Solar", "Oil & Gas", "Electronics",
  ],
  "Lifestyle": [
    "Parenting", "Marriage", "Dating", "Pets", "Astrology", "Religion", "Hobbies",
    "Crafts", "Gardening", "Lifestyle",
  ],
  "Adult": [
    "Adult", "Escort Services", "Companion Services", "Dating Services",
    "Webcam Platforms", "Adult Toys", "Sexual Wellness", "Adult Education",
    "Relationship Advice", "Nightlife",
  ],
  "Local Business": [
    "Local Business", "Plumber", "Electrician", "Carpenter", "Painter",
    "Cleaning Services", "Pest Control", "Laundry", "Repair Services", "Home Services",
  ],
  "Global Brands": [
    "Brand", "Product", "Service", "Company", "Organization", "Government Agency",
    "University", "Non-Profit", "Mobile App", "Website",
  ],
}

_CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
  "Technology": (
    "software", "developer", "development", "app", "api", "cloud", "digital", "tech",
    "platform", "code", "programming", "saas", "startup tech",
  ),
  "Healthcare & Medical": (
    "medical", "health", "patient", "clinical", "doctor", "treatment", "care", "hospital",
    "diagnosis", "therapy", "pharma", "wellness clinic",
  ),
  "Beauty & Fashion": (
    "beauty", "cosmetic", "makeup", "skincare", "fashion", "style", "grooming", "salon",
    "lipstick", "fragrance", "apparel",
  ),
  "Food": (
    "food", "restaurant", "menu", "recipe", "dining", "cuisine", "meal", "cafe", "bakery",
    "grocery", "catering", "beverage",
  ),
  "Automotive": (
    "car", "vehicle", "auto", "motor", "drive", "garage", "dealership", "bike", "ev",
    "automobile", "taxi", "rental car",
  ),
  "Travel": (
    "travel", "trip", "tour", "vacation", "holiday", "booking", "hotel", "flight",
    "resort", "tourism", "visa", "trekking",
  ),
  "Real Estate": (
    "property", "real estate", "home", "apartment", "rent", "buy", "villa", "construction",
    "interior", "furniture", "commercial property",
  ),
  "Finance": (
    "finance", "bank", "invest", "loan", "insurance", "money", "stock", "mutual fund",
    "credit card", "tax", "accounting",
  ),
  "Education": (
    "education", "school", "course", "learning", "training", "student", "college",
    "university", "coaching", "certification", "edtech",
  ),
  "Business": (
    "business", "company", "startup", "consulting", "marketing", "brand", "agency",
    "b2b", "enterprise", "corporate",
  ),
  "Ecommerce": (
    "ecommerce", "online shop", "store", "retail", "marketplace", "product", "dropshipping",
    "wholesale", "shipping", "logistics",
  ),
  "Entertainment": (
    "movie", "music", "game", "stream", "entertainment", "show", "ott", "gaming",
    "celebrity", "photography", "esports",
  ),
  "Sports": (
    "sport", "fitness", "gym", "training", "athlete", "match", "cricket", "football",
    "tennis", "running", "cycling",
  ),
  "Government & Legal": (
    "legal", "law", "government", "court", "visa", "policy", "immigration", "passport",
    "ngo", "charity", "public service",
  ),
  "Industrial": (
    "manufacturing", "factory", "industrial", "machinery", "supply", "mining",
    "agriculture", "solar", "renewable", "chemical",
  ),
  "Lifestyle": (
    "lifestyle", "family", "hobby", "home", "living", "parenting", "marriage", "pets",
    "astrology", "gardening", "crafts",
  ),
  "Adult": (
    "adult", "nightlife", "relationship", "wellness", "dating advice",
  ),
  "Local Business": (
    "near me", "local", "plumber", "electrician", "carpenter", "painter", "cleaning",
    "pest control", "laundry", "repair", "home service", "handyman",
  ),
  "Global Brands": (
    "brand", "product", "service", "company", "organization", "official", "website",
    "mobile app", "non-profit", "government agency",
  ),
}

# Per-domain hint overrides (longer phrases first during scoring)
_DOMAIN_EXTRA_HINTS: dict[str, tuple[str, ...]] = {
  "Technology": ("information technology", "it services", "tech company"),
  "Software": ("software company", "software development", "custom software", "enterprise software"),
  "SaaS": ("saas platform", "subscription software", "cloud software", "b2b saas"),
  "Artificial Intelligence": ("artificial intelligence", " ai ", "ai solutions", "ai platform", "generative ai"),
  "Machine Learning": ("machine learning", " ml ", "model training", "ml pipeline"),
  "Deep Learning": ("deep learning", "neural network", "cnn", "transformer model"),
  "Generative AI": ("generative ai", "llm", "chatgpt", "text generation", "image generation"),
  "Data Science": ("data science", "data analyst", "analytics", "data pipeline"),
  "Big Data": ("big data", "hadoop", "spark", "data warehouse"),
  "Cloud Computing": ("cloud computing", "aws", "azure", "gcp", "cloud migration"),
  "Cybersecurity": ("cybersecurity", "infosec", "penetration test", "security audit"),
  "Blockchain": ("blockchain", "smart contract", "distributed ledger"),
  "Cryptocurrency": ("cryptocurrency", "bitcoin", "crypto exchange", "altcoin"),
  "Web3": ("web3", "decentralized", "dapp", "nft"),
  "Mobile App Development": ("mobile app", "flutter", "react native", "ios app", "android app"),
  "Web Development": ("web development", "website development", "frontend", "full stack"),
  "UI/UX Design": ("ui ux", "user experience", "interface design", "figma"),
  "DevOps": ("devops", "ci cd", "kubernetes", "docker", "infrastructure as code"),
  "Networking": ("networking", "cisco", "lan wan", "network engineer"),
  "IoT": ("iot", "internet of things", "smart device", "embedded"),
  "Healthcare": ("healthcare", "medical", "hospital", "patient care", "ehr", "hipaa"),
  "Hospital": ("hospital", "inpatient", "emergency room", "medical center"),
  "Clinic": ("clinic", "outpatient", "medical clinic", "polyclinic"),
  "Physiotherapy": ("physiotherapy", "physical therapy", "rehabilitation"),
  "Dentist": ("dentist", "dental clinic", "orthodontist", "teeth"),
  "Pharmacy": ("pharmacy", "pharmacist", "prescription", "drugstore"),
  "Laboratory": ("laboratory", "diagnostic lab", "pathology lab", "blood test"),
  "Mental Health": ("mental health", "psychologist", "counseling", "therapy"),
  "Veterinary": ("veterinary", "vet clinic", "pet hospital", "animal doctor"),
  "Medical Equipment": ("medical equipment", "medical devices", "hospital equipment"),
  "Telemedicine": ("telemedicine", "virtual care", "online doctor", "remote consultation"),
  "Diagnostics": ("diagnostics", "medical testing", "imaging", "lab diagnostics"),
  "Nutrition": ("nutrition", "dietitian", "diet plan", "nutritional"),
  "Fitness": ("fitness", "gym training", "workout", "personal trainer"),
  "Yoga": ("yoga", "yoga class", "meditation", "asana"),
  "Wellness": ("wellness", "holistic health", "spa wellness", "self care"),
  "Beauty": ("beauty", "beauty brand", "beauty products", "beauty company"),
  "Cosmetics": ("cosmetics", "makeup", "lipstick", "foundation"),
  "Skincare": ("skincare", "serum", "moisturizer", "anti aging"),
  "Haircare": ("haircare", "shampoo", "hair salon", "hair treatment"),
  "Makeup": ("makeup", "cosmetics", "lip color", "beauty kit"),
  "Perfume": ("perfume", "fragrance", "cologne", "eau de parfum"),
  "Fashion": ("fashion", "apparel", "clothing brand", "style"),
  "Clothing": ("clothing", "apparel", "garments", "wardrobe"),
  "Jewelry": ("jewelry", "jewellery", "gold ring", "diamond"),
  "Watches": ("watches", "luxury watch", "smartwatch", "timepiece"),
  "Footwear": ("footwear", "shoes", "sneakers", "sandals"),
  "Accessories": ("accessories", "handbag", "belt", "sunglasses"),
  "Restaurant": ("restaurant", "dining", "fine dining", "food menu"),
  "Cafe": ("cafe", "coffee shop", "espresso", "bakery cafe"),
  "Bakery": ("bakery", "pastry", "bread", "cake shop"),
  "Fast Food": ("fast food", "quick service", "burger", "pizza"),
  "Food Delivery": ("food delivery", "order food online", "delivery app"),
  "Grocery": ("grocery", "supermarket", "grocery store", "provisions"),
  "Catering": ("catering", "event catering", "wedding catering"),
  "Beverage": ("beverage", "drinks", "juice", "soft drink"),
  "Organic Food": ("organic food", "organic produce", "natural food"),
  "Automobile": ("automobile", "car dealer", "new car"),
  "Luxury Cars": ("luxury cars", "premium car", "bmw", "mercedes"),
  "Used Cars": ("used cars", "pre owned", "second hand car"),
  "Bikes": ("bikes", "motorcycle", "two wheeler", "scooter"),
  "EV Vehicles": ("electric vehicle", " ev ", "ev charging", "tesla"),
  "Car Rental": ("car rental", "rent a car", "vehicle hire"),
  "Taxi": ("taxi", "cab service", "ride hailing"),
  "Auto Parts": ("auto parts", "spare parts", "car accessories parts"),
  "Car Service": ("car service", "auto repair", "car maintenance"),
  "Car Accessories": ("car accessories", "seat cover", "car audio"),
  "Travel": ("travel", "travel agency", "vacation planning"),
  "Tourism": ("tourism", "tourist", "sightseeing", "destination"),
  "Hotels": ("hotel", "hotel booking", "accommodation", "stay"),
  "Resorts": ("resort", "beach resort", "luxury resort"),
  "Flights": ("flights", "airline", "flight booking", "airfare"),
  "Cruise": ("cruise", "cruise ship", "cruise vacation"),
  "Visa": ("visa", "visa application", "travel visa"),
  "Adventure": ("adventure", "adventure sports", "outdoor adventure"),
  "Trekking": ("trekking", "hiking", "mountain trek"),
  "Camping": ("camping", "campsite", "outdoor camping"),
  "Holiday Packages": ("holiday package", "vacation package", "tour package"),
  "Travel Insurance": ("travel insurance", "trip insurance"),
  "Real Estate": ("real estate", "property agent", "realtor"),
  "Property": ("property", "property for sale", "property listing"),
  "Apartments": ("apartment", "flat", "condo", "rental apartment"),
  "Villas": ("villa", "luxury villa", "holiday villa"),
  "Commercial Property": ("commercial property", "office space", "retail space"),
  "Interior Design": ("interior design", "home interior", "interior decorator"),
  "Architecture": ("architecture", "architect", "building design"),
  "Construction": ("construction", "contractor", "building construction"),
  "Home Decor": ("home decor", "home furnishing", "decoration"),
  "Furniture": ("furniture", "sofa", "furniture store"),
  "Finance": ("finance", "financial services", "fintech"),
  "Banking": ("banking", "bank account", "neobank"),
  "Insurance": ("insurance", "life insurance", "health insurance"),
  "Investment": ("investment", "portfolio", "wealth management"),
  "Stock Market": ("stock market", "share trading", "equity market"),
  "Mutual Funds": ("mutual funds", "sip investment", "fund manager"),
  "Loans": ("loans", "personal loan", "home loan"),
  "Credit Cards": ("credit card", "rewards card", "card application"),
  "Tax": ("tax", "income tax", "tax filing", "gst"),
  "Accounting": ("accounting", "bookkeeping", "chartered accountant"),
  "Education": ("education", "learning", "academic"),
  "School": ("school", "primary school", "high school"),
  "College": ("college", "undergraduate", "campus"),
  "University": ("university", "degree program", "higher education"),
  "Online Learning": ("online learning", "elearning", "virtual classroom"),
  "Coaching": ("coaching", "tutor", "coaching class"),
  "Certification": ("certification", "professional certificate", "accredited course"),
  "EdTech": ("edtech", "learning platform", "online course platform"),
  "Competitive Exams": ("competitive exam", "entrance exam", "test prep"),
  "Language Learning": ("language learning", "learn english", "language course"),
  "Business": ("business", "enterprise", "commercial"),
  "Startup": ("startup", "founder", "venture", "seed funding"),
  "Consulting": ("consulting", "consultant", "advisory"),
  "Marketing": ("marketing", "brand marketing", "campaign"),
  "Digital Marketing": ("digital marketing", "online marketing", "ppc"),
  "SEO": ("seo", "search engine optimization", "keyword ranking"),
  "Content Marketing": ("content marketing", "blog strategy", "content plan"),
  "Email Marketing": ("email marketing", "newsletter", "email campaign"),
  "Branding": ("branding", "brand identity", "logo design"),
  "Advertising": ("advertising", "ad campaign", "media buying"),
  "Ecommerce": ("ecommerce", "online store", "shopify"),
  "Marketplace": ("marketplace", "multi vendor", "seller platform"),
  "Dropshipping": ("dropshipping", "dropship store"),
  "Wholesale": ("wholesale", "bulk supplier", "b2b wholesale"),
  "Retail": ("retail", "retail store", "brick and mortar"),
  "B2B": ("b2b", "business to business", "enterprise sales"),
  "B2C": ("b2c", "direct to consumer", "consumer brand"),
  "Logistics": ("logistics", "supply chain", "freight"),
  "Warehousing": ("warehousing", "warehouse", "fulfillment center"),
  "Shipping": ("shipping", "courier", "parcel delivery"),
  "Movies": ("movies", "film", "cinema", "box office"),
  "TV Shows": ("tv shows", "television series", "streaming show"),
  "Music": ("music", "album", "artist", "concert"),
  "OTT": ("ott", "streaming platform", "video on demand"),
  "Gaming": ("gaming", "video game", "game studio"),
  "Esports": ("esports", "competitive gaming", "tournament"),
  "Streaming": ("streaming", "live stream", "broadcaster"),
  "Celebrity": ("celebrity", "famous personality", "star"),
  "Events": ("events", "event management", "conference"),
  "Photography": ("photography", "photographer", "photo studio"),
  "Sports": ("sports", "athletic", "league"),
  "Cricket": ("cricket", "ipl", "cricket bat"),
  "Football": ("football", "soccer", "premier league"),
  "Basketball": ("basketball", "nba"),
  "Tennis": ("tennis", "wimbledon", "tennis racket"),
  "Badminton": ("badminton", "shuttlecock"),
  "Gym": ("gym", "fitness center", "gym membership"),
  "Bodybuilding": ("bodybuilding", "muscle training", "protein"),
  "Running": ("running", "marathon", "jogging"),
  "Cycling": ("cycling", "bicycle", "bike ride"),
  "Government": ("government", "public sector", "municipal"),
  "Legal": ("legal", "legal services", "litigation"),
  "Law Firm": ("law firm", "attorney", "lawyer"),
  "Court": ("court", "judiciary", "legal case"),
  "Immigration": ("immigration", "immigration lawyer", "work permit"),
  "Passport": ("passport", "passport application"),
  "Public Services": ("public services", "civic services"),
  "NGO": ("ngo", "non governmental", "nonprofit org"),
  "Charity": ("charity", "donation", "fundraising"),
  "Public Policy": ("public policy", "policy research"),
  "Manufacturing": ("manufacturing", "production line", "factory"),
  "Machinery": ("machinery", "industrial equipment"),
  "Chemicals": ("chemicals", "chemical industry"),
  "Mining": ("mining", "mineral extraction"),
  "Agriculture": ("agriculture", "farming", "crop"),
  "Dairy": ("dairy", "milk products", "dairy farm"),
  "Renewable Energy": ("renewable energy", "clean energy", "green power"),
  "Solar": ("solar", "solar panel", "photovoltaic"),
  "Oil & Gas": ("oil and gas", "petroleum", "upstream"),
  "Electronics": ("electronics", "electronic components", "semiconductor"),
  "Parenting": ("parenting", "childcare", "new parent"),
  "Marriage": ("marriage", "wedding", "matrimony"),
  "Dating": ("dating", "dating app", "relationship"),
  "Pets": ("pets", "pet care", "pet shop"),
  "Astrology": ("astrology", "horoscope", "zodiac"),
  "Religion": ("religion", "spiritual", "faith"),
  "Hobbies": ("hobbies", "hobby shop", "leisure activity"),
  "Crafts": ("crafts", "handmade", "diy craft"),
  "Gardening": ("gardening", "landscaping", "plants"),
  "Lifestyle": ("lifestyle", "life coaching", "daily living"),
  "Adult": ("adult content", "mature audience"),
  "Escort Services": ("escort service",),
  "Companion Services": ("companion service",),
  "Dating Services": ("dating service", "matchmaking"),
  "Webcam Platforms": ("webcam platform",),
  "Adult Toys": ("adult toy", "intimate wellness product"),
  "Sexual Wellness": ("sexual wellness", "intimate health", "sexual health education"),
  "Adult Education": ("adult education", "continuing education", "lifelong learning"),
  "Relationship Advice": ("relationship advice", "couples counseling", "communication skills"),
  "Nightlife": ("nightlife", "night club", "evening entertainment"),
  "Local Business": ("local business", "small business local", "shop near me"),
  "Plumber": ("plumber", "plumbing", "pipe repair", "leak fix"),
  "Electrician": ("electrician", "electrical repair", "wiring"),
  "Carpenter": ("carpenter", "woodwork", "furniture repair"),
  "Painter": ("painter", "house painting", "wall painting"),
  "Cleaning Services": ("cleaning service", "house cleaning", "maid service"),
  "Pest Control": ("pest control", "exterminator", "termite"),
  "Laundry": ("laundry", "dry cleaning", "wash and fold"),
  "Repair Services": ("repair service", "appliance repair", "fix service"),
  "Home Services": ("home services", "home maintenance", "handyman"),
  "Brand": ("brand", "brand name", "branding"),
  "Product": ("product", "product line", "product company"),
  "Service": ("service", "service provider", "service company"),
  "Company": ("company", "corporation", "firm"),
  "Organization": ("organization", "organisation", "institution"),
  "Government Agency": ("government agency", "public agency", "department"),
  "Non-Profit": ("non profit", "nonprofit", "ngo"),
  "Mobile App": ("mobile app", "app download", "application"),
  "Website": ("website", "official website", "web portal"),
}

# Nutrition appears twice (IDs 33 healthcare, 58 food) — category-scoped hints
_NUTRITION_HEALTH_HINTS = (
  "clinical nutrition", "dietitian", "medical nutrition", "therapeutic diet",
  "hospital diet", "patient nutrition", "supplement therapy",
)
_NUTRITION_FOOD_HINTS = (
  "food nutrition", "nutritional food", "healthy eating", "meal nutrition",
  "food label", "organic nutrition", "diet food",
)

# Adult domains: informational only — no promotional / explicit keyword generation
ADULT_CATEGORY = "Adult"
ADULT_RESTRICTED_DOMAINS: frozenset[str] = frozenset({
  "Escort Services", "Companion Services", "Webcam Platforms", "Adult Toys", "Nightlife",
})
ADULT_INFORMATIONAL_DOMAINS: frozenset[str] = frozenset({
  "Adult", "Dating Services", "Sexual Wellness", "Adult Education", "Relationship Advice",
})
ADULT_BLOCKED_KEYWORD_TERMS: tuple[str, ...] = (
  "escort agency", "call girl", "xxx", "porn", "pornography", "nude", "nsfw",
  "sex chat", "webcam girl", "adult entertainment", "erotic", "prostitut",
  "hookup site", "sugar daddy", "sugar baby",
)

GLOBAL_BRAND_DOMAINS: frozenset[str] = frozenset({
  "Brand", "Product", "Service", "Company", "Organization",
  "Government Agency", "University", "Non-Profit", "Mobile App", "Website",
})
GLOBAL_BRAND_SEED_SIGNALS: tuple[str, ...] = (
  "brand", "product", "products", "company", "organization", "organisation",
  "service", "services", "official", "website", "mobile app", "app",
)

_DOMAIN_PRIORITY: dict[str, int] = {
  "Beauty": 5, "Cosmetics": 5, "Skincare": 5, "Makeup": 5,
  "Healthcare": 4, "Telemedicine": 4, "Hospital": 4,
  "Artificial Intelligence": 4, "Mobile App Development": 4, "SEO": 4,
  "Plumber": 4, "Electrician": 4, "Local Business": 3,
  "Brand": 2, "Product": 2, "Company": 2, "Service": 2,
  "Ecommerce": 2, "Technology": 1, "Business": 1,
}


def _slug_for(domain: str, domain_id: int, name_count: dict[str, int]) -> str:
  base = re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_")
  if name_count.get(domain.lower(), 0) > 1:
    return f"{base}_{domain_id}"
  return base


def _hints_for(domain: str, category: str, domain_id: int) -> tuple[str, ...]:
  parts = tuple(domain.lower().split())
  base = (domain.lower(), domain.lower().replace(" ", ""), *parts)
  cat = _CATEGORY_HINTS.get(category, ())
  extra = _DOMAIN_EXTRA_HINTS.get(domain, ())

  if domain == "Nutrition" and category == "Healthcare & Medical":
    extra = _NUTRITION_HEALTH_HINTS
  elif domain == "Nutrition" and category == "Food":
    extra = _NUTRITION_FOOD_HINTS

  return tuple(dict.fromkeys(base + cat + extra))


def build_master_registry() -> list[dict[str, Any]]:
  registry: list[dict[str, Any]] = []
  name_count: dict[str, int] = {}
  for category, domains in DOMAIN_CATALOG.items():
    for domain in domains:
      key = domain.lower()
      name_count[key] = name_count.get(key, 0) + 1

  idx = 1
  seen_names: dict[str, int] = {}
  for category, domains in DOMAIN_CATALOG.items():
    for domain in domains:
      key = domain.lower()
      seen_names[key] = seen_names.get(key, 0) + 1
      slug = _slug_for(domain, idx, seen_names)
      flags: dict[str, bool] = {
        "adult": category == ADULT_CATEGORY,
        "adult_restricted": domain in ADULT_RESTRICTED_DOMAINS,
        "adult_informational": domain in ADULT_INFORMATIONAL_DOMAINS,
        "global_brand_meta": domain in GLOBAL_BRAND_DOMAINS,
        "local_business": category == "Local Business",
      }
      registry.append({
        "id": idx,
        "domain": domain,
        "slug": slug,
        "category": category,
        "hints": _hints_for(domain, category, idx),
        "flags": flags,
      })
      idx += 1
  return registry


MASTER_DOMAINS: list[dict[str, Any]] = build_master_registry()
DOMAIN_BY_ID: dict[int, dict[str, Any]] = {d["id"]: d for d in MASTER_DOMAINS}
DOMAIN_BY_SLUG: dict[str, dict[str, Any]] = {d["slug"]: d for d in MASTER_DOMAINS}

# First occurrence wins for duplicate display names (e.g. Nutrition → healthcare ID 33)
DOMAIN_BY_NAME: dict[str, dict[str, Any]] = {}
for _d in MASTER_DOMAINS:
  if _d["domain"] not in DOMAIN_BY_NAME:
    DOMAIN_BY_NAME[_d["domain"]] = _d
ALL_DOMAIN_NAMES: tuple[str, ...] = tuple(d["domain"] for d in MASTER_DOMAINS)
ALL_DOMAIN_SLUGS: tuple[str, ...] = tuple(d["slug"] for d in MASTER_DOMAINS)
DOMAIN_COUNT: int = len(MASTER_DOMAINS)


def resolve_domain(name_or_slug: str, *, category: str | None = None) -> dict[str, Any] | None:
  if name_or_slug in DOMAIN_BY_SLUG:
    return DOMAIN_BY_SLUG[name_or_slug]
  if category:
    for entry in MASTER_DOMAINS:
      if entry["domain"] == name_or_slug and entry["category"] == category:
        return entry
  return DOMAIN_BY_NAME.get(name_or_slug)


def is_adult_domain(domain: str) -> bool:
  entry = resolve_domain(domain)
  return bool(entry and entry.get("flags", {}).get("adult"))


def is_adult_restricted(domain: str) -> bool:
  entry = resolve_domain(domain)
  return bool(entry and entry.get("flags", {}).get("adult_restricted"))


def _hint_score(haystack: str, hints: tuple[str, ...]) -> int:
  score = 0
  for h in sorted(hints, key=len, reverse=True):
    if len(h) < 3:
      continue
    if h in haystack:
      score += 2 if len(h) > 8 else 1
  return score


def _nutrition_disambig_bonus(domain: str, category: str, haystack: str) -> int:
  if domain != "Nutrition":
    return 0
  health_signals = ("dietitian", "clinical", "medical", "patient", "hospital", "therapy", "supplement")
  food_signals = ("food", "recipe", "restaurant", "meal", "organic food", "grocery", "cafe")
  if category == "Healthcare & Medical":
    return sum(2 for s in health_signals if s in haystack)
  return sum(2 for s in food_signals if s in haystack)


_VERTICAL_CATEGORIES: frozenset[str] = frozenset(
  c for c in DOMAIN_CATALOG if c not in ("Global Brands",)
)


def _vertical_boost(entry: dict[str, Any], haystack: str) -> int:
  cat = entry["category"]
  bonus = 0
  if cat == "Beauty & Fashion" and any(s in haystack for s in ("beauty", "cosmetic", "makeup", "skincare", "lipstick")):
    bonus += 5
  if cat == "Healthcare & Medical" and any(s in haystack for s in ("healthcare", "medical", "hospital", "patient", "clinic")):
    bonus += 5
  if cat == "Technology" and any(s in haystack for s in ("software", "development", "app", "flutter", "api")):
    bonus += 4
  if cat == "Local Business" and any(s in haystack for s in ("near me", "plumber", "electrician", "repair")):
    bonus += 4
  return bonus


def _global_brand_bonus(domain: str, haystack: str) -> int:
  if domain not in GLOBAL_BRAND_DOMAINS:
    return 0
  # Meta domains — lower weight so verticals win when both match
  return min(2, sum(1 for s in GLOBAL_BRAND_SEED_SIGNALS if s in haystack))


def _local_business_bonus(domain: str, haystack: str) -> int:
  if domain not in DOMAIN_CATALOG.get("Local Business", ()):
    return 0
  bonus = 0
  if "near me" in haystack or "local" in haystack:
    bonus += 2
  return bonus + _hint_score(haystack, _DOMAIN_EXTRA_HINTS.get(domain, ()))


def classify_domains(seed: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
  """Score all 210 domains — domain-first classification with disambiguation."""
  ctx = context or {}
  haystack = f" {seed.lower()} "
  haystack += " " + " ".join(ctx.get("topic_clusters", [])).lower()
  kb_domain = ctx.get("kb_primary_domain")
  kb_slug = ctx.get("kb_primary_domain_slug")
  scored: list[tuple[int, dict[str, Any]]] = []

  for entry in MASTER_DOMAINS:
    domain = entry["domain"]
    hits = _hint_score(haystack, entry["hints"])
    hits += _nutrition_disambig_bonus(domain, entry["category"], haystack)
    hits += _global_brand_bonus(domain, haystack)
    hits += _local_business_bonus(domain, haystack)
    hits += _vertical_boost(entry, haystack)

    if kb_domain and (domain == kb_domain or entry["slug"] == kb_slug):
      hits += 6
    for ent in ctx.get("disambiguated_entities", []):
      if ent.get("domain") == domain or ent.get("domain_slug") == entry["slug"]:
        hits += 4

    if hits > 0:
      scored.append((hits, entry))

  if not scored:
    fallback = DOMAIN_BY_NAME["Business"]
    scored = [(1, fallback)]

  scored.sort(
    key=lambda x: (x[0], _DOMAIN_PRIORITY.get(x[1]["domain"], 2), -x[1]["id"]),
    reverse=True,
  )

  # Prefer vertical domain over Global Brands meta when both match
  primary_entry = scored[0][1]
  if primary_entry["category"] == "Global Brands" and len(scored) > 1:
    top_score = scored[0][0]
    for score, entry in scored[1:]:
      if entry["category"] in _VERTICAL_CATEGORIES and score >= top_score - 2:
        primary_entry = entry
        break

  matched_entries = [s[1] for s in scored[:8]]
  # Re-rank matched list with primary first
  if matched_entries and matched_entries[0]["id"] != primary_entry["id"]:
    matched_entries = [primary_entry] + [e for e in matched_entries if e["id"] != primary_entry["id"]]

  return {
    "primary_domain": primary_entry["domain"],
    "primary_domain_slug": primary_entry["slug"],
    "primary_domain_id": primary_entry["id"],
    "category": primary_entry["category"],
    "domains": [e["domain"] for e in matched_entries],
    "domain_slugs": [e["slug"] for e in matched_entries],
    "domain_scores": {e["domain"]: s for s, e in scored[:12]},
    "industries": [e["domain"] for e in matched_entries[:5]],
    "primary_industry": primary_entry["domain"],
    "domain": primary_entry["slug"],
    "flags": primary_entry.get("flags", {}),
    "total_domains": DOMAIN_COUNT,
  }
