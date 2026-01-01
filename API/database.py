from collections import defaultdict, Counter
import psycopg2 # Example for PostgreSQL

from collections import Counter, defaultdict

def store_fingerprints(interactionid, fingerprints, filename):
    with psycopg2.connect(host="localhost",
        port="5432",
        database="postgres",
        user="postgres",
        password="pwd") as conn:

        with conn.cursor() as cur:
            data = []
            for h, t in fingerprints:
                packed_hash, offset_ms = convert_fingerprint(h, t)
                # Create a 4-item tuple to match: (hash, interactionid, offsetms, filename)
                data.append((packed_hash, interactionid, offset_ms, filename))
            query = """
                INSERT INTO fingerprints 
                (hash, interactionid, offsetms, filename) 
                VALUES (%s, %s, %s, %s)
            """
            cur.executemany(query, data)
            conn.commit()

def find_match_in_db(fingerprints):
    # 1. Prepare packed hashes for the search and the map
    packed_hashes_to_search = []
    # query_map must map: {packed_int_hash: offset_ms}
    # This allows us to compare DB results to our query clip
    query_map = {} 

    for h, t in fingerprints:
        p_hash, p_offset_ms = convert_fingerprint(h, t)
        packed_hashes_to_search.append(p_hash)
        query_map[p_hash] = p_offset_ms # Store the packed version!

    # 2. Database Connection
    query = "SELECT interactionid, offsetms, hash, filename FROM fingerprints WHERE hash IN %s"
    
    with psycopg2.connect(host="localhost", port="5432", database="postgres", user="postgres", password="pwd") as conn:
        with conn.cursor() as cur:
            # We must pass a tuple of ALL packed hashes
            cur.execute(query, (tuple(packed_hashes_to_search),))
            results = cur.fetchall()

            # 3. Grouping results by interactionid
            matches = defaultdict(lambda: {"offsets": [], "filename": None})
            
            for interactionid, db_offset_ms, h, filename in results:
                # Get the query's offset for this same hash
                q_offset_ms = query_map.get(h) 
                
                if q_offset_ms is None:
                    continue 
                
                # Calculate the relative time difference (Delta)
                # Since these are integers (ms), no rounding is needed here
                delta_t = db_offset_ms - q_offset_ms
                
                if matches[interactionid]["filename"] is None:
                    matches[interactionid]["filename"] = filename
                
                matches[interactionid]["offsets"].append(delta_t)

            # 4. Scoring and Alignment
            final_results = []
            for interactionid, data in matches.items():
                if not data["offsets"]:
                    continue
                
                # Use Counter to find the most frequent Delta (the alignment peak)
                counts = Counter(data["offsets"])
                best_offset, hit_count = counts.most_common(1)[0]
                
                # Score = how many hashes lined up at that specific offset
                # len(fingerprints) is the total number of hashes we searched for
                score = hit_count / len(fingerprints)
                
                final_results.append({
                    "interactionid": interactionid,
                    "score": round(score, 4),
                    "offset_ms": best_offset,
                    "filename": data["filename"]
                })

            # Sort by highest score first
            return sorted(final_results, key=lambda x: x['score'], reverse=True)

def convert_fingerprint(hash_str, timestamp):
    """
    Converts 'b9_b11' and 0.123456 into (packed_int, offset_ms)
    """
    # 1. Clean the string (remove 'b' prefixes if they exist)
    clean_hash = hash_str.replace('b', '')
    bin1, bin2 = map(int, clean_hash.split('_'))
    
    # 2. Pack into 32-bit integer
    # This remains the fastest way for SQL to index the data
    packed_hash = (bin1 << 16) | bin2
    
    # 3. Handle Microseconds/Drift
    # Rounding to the nearest 10ms (0.01s) is a common trick. 
    # It's 'fuzzy' enough to ignore micro-jitters but precise enough for alignment.
    offset_ms = int(round(timestamp, 2) * 1000)
    print(f"packed_hash, offset_ms={packed_hash}, {offset_ms}")
    return packed_hash, offset_ms

def align_and_score(query_fps, db_results, time_tolerance_ms=200):
    """
    query_fps: The fingerprints from 10s search clip
    db_results: The rows found in SQL [(interactionid, offset_ms, hash), ...]
    """
    # 1. Map query hashes to their local timestamps for easy math
    query_map = defaultdict(list)
    for h, t in query_fps:
        query_map[h].append(t)

    # 2. Calculate the 'Delta' for every match
    # Delta = DB_Time - Query_Time
    file_deltas = defaultdict(list)
    for interactionid, db_offset, h in db_results:
        for q_offset in query_map[h]:
            delta = db_offset - q_offset
            file_deltas[interactionid].append(delta)

    # 3. Find the winning File
    results = []
    for interactionid, deltas in file_deltas.items():
        if not deltas: continue
        
        # We look for the most common delta (the 'Peak' in the histogram)
        # We use a window (time_tolerance_ms) because of your microsecond drift
        counts = Counter([round(d / 100) * 100 for d in deltas]) # Group by 100ms
        best_delta, hit_count = counts.most_common(1)[0]
        
        # Calculate a final confidence score
        # A score > 0.5 is usually a guaranteed match
        score = hit_count / len(query_fps)
        
        results.append({
            "interactionid": interactionid,
            "score": round(score, 4),
            "best_offset_ms": best_delta
        })

    return sorted(results, key=lambda x: x['score'], reverse=True)