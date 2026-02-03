#!/bin/bash
"""
./run-cn.sh
"""

# 1. Preprocess data
python community-notes/sourcecode/data-processing.py

# 2. Run Community Notes scoring
python community-notes/sourcecode/main.py \
  --enrollment data/userEnrollment-00001.tsv \
  --notes data/notes-00001.tsv \
  --ratings data/ratings-00001.tsv \
  --status data/noteStatusHistory-00001.tsv \
  --outdir data
