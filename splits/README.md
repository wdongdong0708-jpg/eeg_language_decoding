# Split artifacts

Split files are generated from canonical `content_id` values, never from sample
or window indices. Each artifact must record:

- hash algorithm and seed;
- train/valid/test fractions;
- manifest fingerprint;
- content-ID version;
- integrity-check results;
- optional held-out subject set.

Generated split files are ignored by Git until a reviewed version is selected
for release.

