# OCR fixtures

`plain.png` is a synthetic 360 × 110 RGB PNG with the literal text **A7B9C2**
drawn in black Arial Bold (64 px) on white. It was generated locally with Pillow;
no font file or third-party CAPTCHA corpus is distributed here.

The expected text was chosen before running OCR. This fixture verifies a clean,
single-line baseline and file/byte API equivalence. It does not measure accuracy
on distorted CAPTCHA images.

`invoice.png` and `cheque.png` are synthetic 1250×750 RGB images rendered locally
in Arial. They contain fictional printed English labels; the cheque is marked
**SAMPLE - NOT VALID FOR PAYMENT**. No real account, signature or bank document
is reproduced. `scripts/generate_document_fixtures.py` records the text and font
settings; no font binaries are distributed.

- Invoice: total 115.50, reference 0007. 40,914 bytes, SHA-256
  `b14255f49b615a1a71c806098d9d8abb49a4bc8227cf956c7c7c508d045bee71`.
- Cheque: amount 1234.56, number 000042. 36,898 bytes, SHA-256
  `2dadb966c5396b8d7d26290465d28968c846833873a96d9d3b1db0960969db1f`.

These verify printed-document OCR integration and exact amount parsing, not
handwriting, MICR, authenticity or real-world financial-document accuracy.
