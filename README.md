# audioidentificationshazam

# different strategies for comparing fingerprints

1. "jaccard"
Treats the fingerprints as sets (ignores duplicate hashes and timing).

Uses the Jaccard similarity:

J(A,B)= ∣A∪B∣ / ∣A∩B∣
​

Good when you only care if the two audios share the same content features, regardless of order or repetitions.

⚠️ Weak for noisy data, because duplicates (like repeating beats) are discarded.

2. "multiset"
Like "jaccard", but treats fingerprints as multisets (keeps duplicates).

Compares the count of common hashes relative to the minimum total count.

Example: if a peak hash appears 10 times in one file and 12 times in the other, it counts as 10 matches.

Better for repetitive audio (beats, rhythms).

3. "time"
Compares both the hash and timing offsets.

Idea: in Shazam-style fingerprinting, it’s not enough that two hashes exist — they must appear at approximately the same time difference in both audios.

Example: if fingerprint (hash=0x1234, t=12.3s) in file A matches (hash=0x1234, t=12.4s) in file B (within tolerance), it counts as a match.

Much more robust for actual audio matching.

4. "auto"
Automatically decides which method to use:

If your fingerprints are simple hashes → defaults to "jaccard".

If fingerprints include (hash, time) tuples → defaults to "time".

A convenience option so you don’t have to think about the right method.

✅ Rule of thumb:

Use "time" if your fingerprints include (hash, time) pairs (recommended for Shazam-style).

Use "jaccard" if you only have plain hash lists.

Use "multiset" if duplicates carry meaning (e.g., rhythmic signals).

"auto" is just a fallback.

Comparison
 Full audio with audio combining Part1 and Part2 into single audio(39 seconds) - "similarity_score": 0, "is_match": false
 Fullaudio(49 seconds) with Part1(first 10 seconds) - "similarity_score": 0.271889400921659, "is_match": false
 Fullaudio with muted 10 seconds audio - "similarity_score": 0.9496567505720824,"is_match": true
 Fullaudio with Part2(last 29 seconds) - "similarity_score": 0,"is_match": false