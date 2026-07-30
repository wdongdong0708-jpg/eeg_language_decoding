# Metadata

This directory stores small, reviewable metadata needed to make implicit
dataset conventions explicit, for example:

- subject-to-audio-speaker mapping;
- GarnettDream f1/m1 material variants;
- canonical book/chapter/sentence identifiers;
- exclusions and their evidence;
- manual alignment corrections with reviewer and provenance.

Generated metadata belongs under `metadata/generated/` and is ignored by Git.
Every manual table must include a schema/version field and a provenance column.

