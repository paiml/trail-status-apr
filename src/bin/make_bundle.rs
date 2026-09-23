//! One-time conversion: the 3 separate artifacts
//! (blankets_creek_logreg.apr, rope_mill_logreg.apr, tfidf_vocab.tsv)
//! produced by `trail_status_classical.rs` -> one self-contained
//! `trail_status_model.apr`.
//!
//! Usage:
//!   make-bundle <models_dir> <output.apr>
//!   (expects <models_dir>/{blankets_creek_logreg.apr,rope_mill_logreg.apr,tfidf_vocab.tsv})

use aprender::classification::LogisticRegression;
use aprender::format::{load, save, Compression, ModelType, SaveOptions};
use std::collections::HashMap;
use trail_status_infer::bundle::TrailStatusBundle;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: make-bundle <models_dir> <output.apr>");
        std::process::exit(1);
    }
    let models_dir = &args[1];
    let output_path = &args[2];

    let blankets_clf: LogisticRegression =
        load(format!("{models_dir}/blankets_creek_logreg.apr"), ModelType::LogisticRegression)
            .expect("load blankets model");
    let rope_clf: LogisticRegression =
        load(format!("{models_dir}/rope_mill_logreg.apr"), ModelType::LogisticRegression)
            .expect("load rope model");

    let vocab_tsv = std::fs::read_to_string(format!("{models_dir}/tfidf_vocab.tsv"))
        .expect("read tfidf_vocab.tsv");
    let mut vocabulary = HashMap::new();
    let mut idf = Vec::new();
    for line in vocab_tsv.lines() {
        let mut parts = line.splitn(3, '\t');
        let word = parts.next().expect("word column").to_string();
        let idx: usize = parts.next().expect("index column").parse().expect("index is int");
        let idf_val: f64 = parts.next().expect("idf column").parse().expect("idf is float");
        vocabulary.insert(word, idx);
        if idx >= idf.len() {
            idf.resize(idx + 1, 0.0);
        }
        idf[idx] = idf_val;
    }
    println!("Loaded vocabulary: {} terms", vocabulary.len());

    let bundle = TrailStatusBundle { vocabulary, idf, ngram_range: (1, 2), blankets_clf, rope_clf };

    let save_opts = SaveOptions { compression: Compression::None, ..SaveOptions::default() };
    save(&bundle, ModelType::NeuralCustom, output_path, save_opts).expect("save bundle");

    let size = std::fs::metadata(output_path).expect("stat output").len();
    println!("Saved single-file bundle: {output_path} ({size} bytes)");

    // Round-trip check.
    let reloaded: TrailStatusBundle =
        load(output_path, ModelType::NeuralCustom).expect("reload bundle");
    assert_eq!(reloaded.vocabulary.len(), bundle.vocabulary.len());
    assert_eq!(reloaded.idf.len(), bundle.idf.len());
    println!("Round-trip check passed: reloaded bundle has matching vocabulary/IDF sizes.");
}
