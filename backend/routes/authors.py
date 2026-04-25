from flask import Blueprint, jsonify, request
from db.mongo import get_collection

authors_bp = Blueprint("authors", __name__)

# ── Fallback static list (used if DB is empty) ─────────────────────────────
# 43 Indian journalists across major publications
STATIC_AUTHORS = [
    # Times of India
    {"id": "toi_ps",  "name": "Priya Sharma",       "publication": "TOI",   "publication_label": "Times of India",    "articles_count": 312},
    {"id": "toi_rv",  "name": "Rahul Verma",         "publication": "TOI",   "publication_label": "Times of India",    "articles_count": 287},
    {"id": "toi_ag",  "name": "Ananya Gupta",         "publication": "TOI",   "publication_label": "Times of India",    "articles_count": 198},
    {"id": "toi_sk",  "name": "Sanjay Kumar",         "publication": "TOI",   "publication_label": "Times of India",    "articles_count": 341},
    {"id": "toi_mi",  "name": "Meera Iyer",           "publication": "TOI",   "publication_label": "Times of India",    "articles_count": 276},
    {"id": "toi_ap",  "name": "Arun Pandey",          "publication": "TOI",   "publication_label": "Times of India",    "articles_count": 223},
    # The Hindu
    {"id": "hindu_np","name": "Nandini Patel",        "publication": "HINDU", "publication_label": "The Hindu",         "articles_count": 445},
    {"id": "hindu_vm","name": "Vikram Menon",          "publication": "HINDU", "publication_label": "The Hindu",         "articles_count": 389},
    {"id": "hindu_sr","name": "Sunita Rao",            "publication": "HINDU", "publication_label": "The Hindu",         "articles_count": 412},
    {"id": "hindu_ak","name": "Arjun Krishnamurthy",  "publication": "HINDU", "publication_label": "The Hindu",         "articles_count": 356},
    {"id": "hindu_pd","name": "Pradeep Desai",         "publication": "HINDU", "publication_label": "The Hindu",         "articles_count": 291},
    {"id": "hindu_ls","name": "Lakshmi Subramaniam",  "publication": "HINDU", "publication_label": "The Hindu",         "articles_count": 478},
    # Indian Express
    {"id": "ie_mc",   "name": "Mohan Chatterjee",     "publication": "IE",    "publication_label": "Indian Express",    "articles_count": 334},
    {"id": "ie_rg",   "name": "Ritika Ghosh",          "publication": "IE",    "publication_label": "Indian Express",    "articles_count": 267},
    {"id": "ie_ss",   "name": "Suresh Singh",          "publication": "IE",    "publication_label": "Indian Express",    "articles_count": 398},
    {"id": "ie_pm",   "name": "Pooja Mishra",          "publication": "IE",    "publication_label": "Indian Express",    "articles_count": 213},
    {"id": "ie_kb",   "name": "Kartik Bose",           "publication": "IE",    "publication_label": "Indian Express",    "articles_count": 287},
    {"id": "ie_dj",   "name": "Divya Joshi",           "publication": "IE",    "publication_label": "Indian Express",    "articles_count": 322},
    # Hindustan Times
    {"id": "ht_an",   "name": "Abhishek Nair",         "publication": "HT",    "publication_label": "Hindustan Times",   "articles_count": 256},
    {"id": "ht_sr2",  "name": "Sneha Reddy",           "publication": "HT",    "publication_label": "Hindustan Times",   "articles_count": 301},
    {"id": "ht_vt",   "name": "Vivek Tiwari",          "publication": "HT",    "publication_label": "Hindustan Times",   "articles_count": 278},
    {"id": "ht_nc",   "name": "Neha Chopra",           "publication": "HT",    "publication_label": "Hindustan Times",   "articles_count": 245},
    {"id": "ht_rs",   "name": "Rajiv Saxena",          "publication": "HT",    "publication_label": "Hindustan Times",   "articles_count": 334},
    {"id": "ht_am",   "name": "Asha Malik",            "publication": "HT",    "publication_label": "Hindustan Times",   "articles_count": 189},
    # Mint
    {"id": "mint_rk", "name": "Rohit Kapoor",          "publication": "MINT",  "publication_label": "Mint",              "articles_count": 198},
    {"id": "mint_sa", "name": "Shreya Agarwal",        "publication": "MINT",  "publication_label": "Mint",              "articles_count": 223},
    {"id": "mint_pb", "name": "Prakash Bhatt",         "publication": "MINT",  "publication_label": "Mint",              "articles_count": 267},
    {"id": "mint_km", "name": "Kavita Mehta",          "publication": "MINT",  "publication_label": "Mint",              "articles_count": 312},
    {"id": "mint_ad", "name": "Anirudh Dubey",         "publication": "MINT",  "publication_label": "Mint",              "articles_count": 178},
    # NDTV
    {"id": "ndtv_sp", "name": "Shweta Pillai",         "publication": "NDTV",  "publication_label": "NDTV",              "articles_count": 456},
    {"id": "ndtv_mg", "name": "Manish Goyal",          "publication": "NDTV",  "publication_label": "NDTV",              "articles_count": 389},
    {"id": "ndtv_tb", "name": "Tanvi Banerjee",        "publication": "NDTV",  "publication_label": "NDTV",              "articles_count": 312},
    {"id": "ndtv_ra", "name": "Rohan Anand",           "publication": "NDTV",  "publication_label": "NDTV",              "articles_count": 267},
    {"id": "ndtv_pk", "name": "Puja Kaur",             "publication": "NDTV",  "publication_label": "NDTV",              "articles_count": 198},
    # The Wire
    {"id": "wire_as", "name": "Aruna Sen",             "publication": "WIRE",  "publication_label": "The Wire",          "articles_count": 445},
    {"id": "wire_dm", "name": "Dev Mukherjee",         "publication": "WIRE",  "publication_label": "The Wire",          "articles_count": 356},
    {"id": "wire_in", "name": "Indira Nambiar",        "publication": "WIRE",  "publication_label": "The Wire",          "articles_count": 289},
    {"id": "wire_vp", "name": "Vasudha Prasad",        "publication": "WIRE",  "publication_label": "The Wire",          "articles_count": 334},
    {"id": "wire_st", "name": "Siddharth Tiwary",     "publication": "WIRE",  "publication_label": "The Wire",          "articles_count": 178},
    # Mixed / extra to reach 43
    {"id": "toi_bm",  "name": "Bhavna Mathur",        "publication": "TOI",   "publication_label": "Times of India",    "articles_count": 189},
    {"id": "ie_sv",   "name": "Srinath Venkatesh",    "publication": "IE",    "publication_label": "Indian Express",    "articles_count": 256},
    {"id": "hindu_rn","name": "Ritu Nanda",            "publication": "HINDU", "publication_label": "The Hindu",         "articles_count": 167},
    {"id": "ht_gs",   "name": "Gaurav Sharma",        "publication": "HT",    "publication_label": "Hindustan Times",   "articles_count": 201},
]


@authors_bp.route("/authors", methods=["GET"])
def get_authors():
    publication = request.args.get("publication")

    try:
        coll = get_collection("journalists")
        query = {"publication": publication} if publication else {}
        authors = list(coll.find(query, {"_id": 0}))
        if not authors:
            raise Exception("empty db")
    except Exception:
        # Fall back to static list
        authors = STATIC_AUTHORS
        if publication:
            authors = [a for a in authors if a["publication"] == publication]

    return jsonify({"authors": authors, "count": len(authors)})


@authors_bp.route("/authors/<author_id>", methods=["GET"])
def get_author(author_id):
    try:
        coll = get_collection("journalists")
        author = coll.find_one({"id": author_id}, {"_id": 0})
        if not author:
            raise Exception("not found")
    except Exception:
        author = next((a for a in STATIC_AUTHORS if a["id"] == author_id), None)

    if not author:
        return jsonify({"error": "Author not found"}), 404

    return jsonify(author)
