import sqlite3
from collections import Counter

def create_database():
    conn = sqlite3.connect("fingerprints.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS fingerprints (
            hash TEXT,
            offset INTEGER,
            song_id TEXT
        )
    ''')
    conn.commit()
    conn.close()

# def store_fingerprints(fingerprints, song_id):
#     conn = sqlite3.connect("fingerprints.db")
#     c = conn.cursor()
#     for h, t in fingerprints:
#         c.execute("INSERT INTO fingerprints (hash, offset, song_id) VALUES (?, ?, ?)", (h, t, song_id))
#     conn.commit()
#     conn.close()

def store_fingerprints(fingerprints, song_id):
    conn = sqlite3.connect("fingerprints.db")
    c = conn.cursor()
    for h, t in fingerprints:
        c.execute("INSERT INTO fingerprints (hash, offset, song_id) VALUES (?, ?, ?)", (h, int(t), song_id))
    conn.commit()
    conn.close()

def match_fingerprints(fingerprints):
    conn = sqlite3.connect("fingerprints.db")
    c = conn.cursor()
    matches = []
    for h, t in fingerprints:
        c.execute("SELECT offset, song_id FROM fingerprints WHERE hash = ?", (h,))
        for db_offset, song_id in c.fetchall():
            time_diff = int(db_offset) - t  # ✅ Fix: cast to int
            matches.append((song_id, time_diff))
    conn.close()
    if not matches:
        return ("No match", 0)
    result = Counter([m[0] for m in matches])
    return result.most_common(1)[0]


# def match_fingerprints(fingerprints):
#     conn = sqlite3.connect("fingerprints.db")
#     c = conn.cursor()
#     matches = []
#     for h, t in fingerprints:
#         c.execute("SELECT offset, song_id FROM fingerprints WHERE hash = ?", (h,))
#         for db_offset, song_id in c.fetchall():
#             time_diff = db_offset - t
#             matches.append((song_id, time_diff))
#     conn.close()
#     if not matches:
#         return ("No match", 0)
#     result = Counter([m[0] for m in matches])
#     return result.most_common(1)[0]
