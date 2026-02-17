-- 1. Create the Parent Table (Partitioned by HASH)
-- We partition by 'fingerprint_hash' so lookups only hit 1/N partitions.
CREATE TABLE fingerprints (
    hash INT NOT NULL,  -- Bit-packed freq bins (e.g., 9 and 11)
    id INT NOT NULL,           -- Unique ID of the audio file
    timeoffset INT NOT NULL          -- timeoffset in milliseconds (saves space vs float)
) PARTITION BY HASH (hash);

-- 2. Create the Partitions (Example: 4 partitions)
-- For Petabytes, you might create 128 or 256 partitions.
CREATE TABLE fingerprints_p0 PARTITION OF fingerprints FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE fingerprints_p1 PARTITION OF fingerprints FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE fingerprints_p2 PARTITION OF fingerprints FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE fingerprints_p3 PARTITION OF fingerprints FOR VALUES WITH (MODULUS 4, REMAINDER 3);

-- 3. The Covering Index
-- We use a composite index: (hash, file_id, timeoffset)
-- This allows "Index-Only Scans," meaning the DB never has to look at the actual 
-- table rows to answer your query.
CREATE INDEX idx_fingerprints_lookup 
ON fingerprints (hash, id, timeoffset);