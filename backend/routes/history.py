import time
from flask import Blueprint, jsonify, request
from db.mongo import get_collection

history_bp = Blueprint("history", __name__)


@history_bp.route("/history", methods=["GET"])
def list_sessions():
    user_id = request.args.get("user_id", "anonymous")
    limit   = min(int(request.args.get("limit", 30)), 100)

    try:
        coll = get_collection("chat_sessions")
        cursor = (
            coll.find({"user_id": user_id}, {"_id": 0, "messages": 0})
                .sort("updated_at", -1)
                .limit(limit)
        )
        sessions = []
        for doc in cursor:
            sessions.append({
                "id":               doc["session_id"],
                "author_id":        doc.get("author_id", ""),
                "author_name":      doc.get("author_name", doc.get("author_id", "")),
                "publication":      doc.get("publication", ""),
                "publication_code": doc.get("publication", "")[:4].upper(),
                "preview":          doc.get("preview", ""),
                "created_at":       doc.get("created_at", 0),
                "updated_at":       doc.get("updated_at", 0),
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"sessions": sessions})


@history_bp.route("/history/<session_id>", methods=["GET"])
def get_session(session_id):
    try:
        coll = get_collection("chat_sessions")
        doc = coll.find_one({"session_id": session_id}, {"_id": 0})
        if not doc:
            return jsonify({"error": "Session not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Shape messages for frontend
    messages = []
    for m in doc.get("messages", []):
        messages.append({
            "id":         session_id + "_" + str(int(m.get("created_at", 0))),
            "payload": {
                "sourceText":       m["source_text"],
                "publication":      m["publication"],
                "authorId":         m["author_id"],
                "authorName":       doc.get("author_name", m["author_id"]),
            },
            "results":    m["results"],
            "isLoading":  False,
        })

    return jsonify({
        "session_id": session_id,
        "author_id":  doc.get("author_id"),
        "publication": doc.get("publication"),
        "messages":   messages,
    })


@history_bp.route("/history/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    try:
        coll = get_collection("chat_sessions")
        result = coll.delete_one({"session_id": session_id})
        if result.deleted_count == 0:
            return jsonify({"error": "Session not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"deleted": True, "session_id": session_id})
