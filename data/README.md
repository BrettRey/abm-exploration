# Data

This directory contains scripts for downloading external datasets. The data itself is not included in this repository due to size and licensing.

## Scots Syntax Atlas (SCOSYA)

**License:** CC BY-NC-SA 4.0

**Citation:** Smith, Jennifer; Adger, David; Aitken, Brian; Heycock, Caroline; Jamieson, E and Thoms, Gary. 2019. *The Scots Syntax Atlas*. University of Glasgow. https://scotssyntaxatlas.ac.uk

**To download:**

```bash
cd data/scots
python download_all.py --delay 5
```

The API has rate limiting. If blocked, wait 15 minutes and re-run with `--resume`, or email scotssyntaxatlas@gmail.com for bulk access.

**What it contains:** Likert-scale acceptability judgments for 258 syntactic features across 147 Scottish locations (500+ speakers), plus a 275-hour spoken corpus.

## MegaAcceptability v2

**License:** See https://megaattitude.io

**Citation:** White, A. S. and Rawlins, K. 2020. Frequency, acceptability, and selection: A case study of clause-embedding. *Glossa* 5(1), 105.

**To download:**

```bash
cd data/mega
curl -L http://megaattitude.io/projects/mega-acceptability/mega-acceptability-v2.zip -o mega-acceptability-v2.zip
unzip mega-acceptability-v2.zip
```

**What it contains:** 375,000 gradient acceptability ratings (1-7 Likert) for 1,007 English verbs in 50 syntactic frames.
