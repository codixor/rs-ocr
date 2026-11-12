//! Native PP-DocLayout-S regions and baseline/assisted extraction on identical Tiny spans.
use rs_ocr::{
    bills, cheques,
    documents::layout::{LayoutDetector, LayoutDocument, LayoutOptions},
    documents::{
        Amount, DecimalSeparator, Document, DocumentReader, ExtractionOptions, Field, TextSpan,
    },
};
use serde_json::{Value, json};
use std::{
    env,
    error::Error,
    io::{self, BufRead},
};

fn text_field(field: &Field<String>) -> Value {
    json!({"value": field.unique().map(|c| &c.value), "candidates": field.candidates.iter().map(|c| json!({"value":c.value,"raw_text":c.raw_text,"confidence":c.confidence,"span_indices":c.span_indices})).collect::<Vec<_>>()})
}

fn money_field(field: &Field<Amount>) -> Value {
    json!({"value": field.unique().map(|c| c.value.minor_units), "candidates": field.candidates.iter().map(|c| json!({"value":c.value.minor_units,"raw_text":c.raw_text,"confidence":c.confidence,"span_indices":c.span_indices})).collect::<Vec<_>>()})
}

fn process(
    job: &Value,
    reader: &mut Option<DocumentReader>,
    models: &str,
    detector: &LayoutDetector,
) -> Result<Value, Box<dyn Error>> {
    let start = std::time::Instant::now();
    let layout = detector.detect_file(
        job["image"].as_str().ok_or("missing image")?,
        LayoutOptions::default(),
    )?;
    let layout_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = std::time::Instant::now();
    let document = if let Some(spans) = job.get("spans") {
        Document::new(
            spans
                .as_array()
                .ok_or("spans must be an array")?
                .iter()
                .map(|s| {
                    Ok(TextSpan {
                        text: s["text"].as_str().ok_or("missing span text")?.into(),
                        confidence: serde_json::from_value(s["confidence"].clone())?,
                        box_points: serde_json::from_value(s["box_points"].clone())?,
                    })
                })
                .collect::<Result<_, Box<dyn Error>>>()?,
        )?
    } else {
        if reader.is_none() {
            *reader = Some(DocumentReader::from_model_dir(models)?);
        }
        reader
            .as_mut()
            .ok_or("reader unavailable")?
            .read_file(job["image"].as_str().ok_or("missing image")?)?
    };
    let ocr_ms = start.elapsed().as_secs_f64() * 1000.0;
    let companion = LayoutDocument::new(document.clone(), layout)?;
    let regions:Vec<_>=companion.layout().regions.iter().map(|r|json!({"class_id":r.class.code(),"class_name":r.class.name(),"confidence":r.confidence,"bounds":[r.bounds.left,r.bounds.top,r.bounds.right,r.bounds.bottom],"raw_bounds":r.raw_bounds})).collect();
    let spans: Vec<_> = document
        .spans()
        .iter()
        .map(|s| json!({"text":s.text,"confidence":s.confidence,"box_points":s.box_points}))
        .collect();
    let mut options = ExtractionOptions::default();
    match job["decimal_separator"].as_str().unwrap_or("dot") {
        "dot" => {}
        "comma" => options.amount_format.decimal_separator = DecimalSeparator::Comma,
        _ => return Err("expected dot or comma decimal separator".into()),
    }
    let start = std::time::Instant::now();
    let mut versions = Vec::new();
    for assisted in [false, true] {
        let (fields, issues) = match job["kind"].as_str() {
            Some("bill") => {
                let r = if assisted {
                    companion.extract_bill(options)?
                } else {
                    bills::extract(document.clone(), options)?
                };
                (
                    json!({"merchant":text_field(&r.merchant),"date":text_field(&r.date),"reference_number":text_field(&r.reference_number),"currency":text_field(&r.currency),"subtotal":money_field(&r.subtotal),"tax":money_field(&r.tax),"total":money_field(&r.total)}),
                    format!("{:?}", r.issues),
                )
            }
            Some("cheque") => {
                let r = if assisted {
                    companion.extract_cheque(options)?
                } else {
                    cheques::extract(document.clone(), options)?
                };
                (
                    json!({"payee":text_field(&r.payee),"date":text_field(&r.date),"cheque_number":text_field(&r.cheque_number),"bank":text_field(&r.bank),"account_number":text_field(&r.account_number),"amount":money_field(&r.amount),"amount_words":text_field(&r.amount_words),"micr":text_field(&r.micr)}),
                    format!("{:?}", r.issues),
                )
            }
            _ => return Err("expected bill or cheque".into()),
        };
        versions.push(json!({"fields":fields,"issues":issues}));
    }
    Ok(json!({"id":job["id"],"kind":job["kind"],"spans":spans,
        "baseline":versions[0],"assisted":versions[1],"ocr_cached":job.get("spans").is_some(),
        "layout":{"width":companion.layout().width,"height":companion.layout().height,"regions":regions,"region_spans":companion.region_spans()},
        "elapsed_ms":{"layout":layout_ms,"ocr":ocr_ms,"extraction":start.elapsed().as_secs_f64()*1000.0}}))
}

fn main() -> Result<(), Box<dyn Error>> {
    let layout_model = env::args()
        .nth(1)
        .ok_or("usage: doc_layout_batch LAYOUT_ONNX TINY_DIR < jobs.jsonl")?;
    let models = env::args().nth(2).ok_or("missing Tiny directory")?;
    let detector = LayoutDetector::new(layout_model)?;
    let mut reader = None;
    for line in io::stdin().lock().lines() {
        let job: Value = serde_json::from_str(&line?)?;
        let result = process(&job, &mut reader, &models, &detector)
            .unwrap_or_else(|e| json!({"id":job["id"],"error":e.to_string()}));
        println!("{result}");
    }
    Ok(())
}
