//! Shared load + predict logic for the trail-status classifier, used by
//! both the CLI and the API binary. The model -- vocabulary, IDF weights,
//! and both LogisticRegression classifiers, bundled as a single
//! `trail_status_model.apr` (see `bundle.rs`, `bin/make_bundle.rs`) -- is
//! embedded directly into the compiled binary via `include_bytes!` and
//! `aprender::format::load_from_bytes`, aprender's own documented
//! single-binary-deployment mechanism (`apr-format/src/core_io.rs:149`:
//! "Enables the include_bytes!() pattern for embedding models directly in
//! executables"). No `.apr` file is required on disk at runtime -- the
//! CLI and API binaries are fully self-contained once built.
//!
//! Trade-off: the model is now fixed at compile time. Updating it means
//! rebuilding (`cargo build --release`), not swapping a file next to the
//! binary.

use crate::bundle::TrailStatusBundle;
use aprender::format::{load_from_bytes, ModelType};
use aprender::primitives::Matrix;

/// Baked in at compile time from `models/trail_status_model.apr`, relative
/// to this source file. Rebuilding after `make-bundle` regenerates that
/// file picks up the new model automatically.
static MODEL_BYTES: &[u8] = include_bytes!("../models/trail_status_model.apr");

pub struct TrailStatusModel {
    bundle: TrailStatusBundle,
}

#[derive(Debug, serde::Serialize)]
pub struct Prediction {
    pub blankets_creek_open: bool,
    pub blankets_creek_confidence: f32,
    pub rope_mill_open: bool,
    pub rope_mill_confidence: f32,
}

impl TrailStatusModel {
    /// Loads the model embedded in this binary. Takes no arguments and
    /// cannot fail on a missing file -- the bytes are part of the binary.
    pub fn load() -> Self {
        let bundle: TrailStatusBundle = load_from_bytes(MODEL_BYTES, ModelType::NeuralCustom)
            .expect("embedded trail-status bundle is corrupt (checksum failure) -- this indicates a build problem, not a runtime one");
        Self { bundle }
    }

    pub fn predict(&self, text: &str) -> Prediction {
        let features = tfidf_transform(
            text,
            &self.bundle.vocabulary,
            &self.bundle.idf,
            self.bundle.ngram_range,
        );
        let x = Matrix::from_vec(1, features.len(), features).expect("shape matches vocab size");

        let pb = self.bundle.blankets_clf.predict_proba(&x).as_slice()[0];
        let pr = self.bundle.rope_clf.predict_proba(&x).as_slice()[0];

        Prediction {
            blankets_creek_open: pb >= 0.5,
            blankets_creek_confidence: if pb >= 0.5 { pb } else { 1.0 - pb },
            rope_mill_open: pr >= 0.5,
            rope_mill_confidence: if pr >= 0.5 { pr } else { 1.0 - pr },
        }
    }
}

/// Hand-rolled TF-IDF transform against a pre-built vocabulary, matching
/// `aprender::text::vectorize::TfidfVectorizer`'s own formula exactly:
/// whitespace tokenize + lowercase (`WhitespaceTokenizer`), 1..=2-gram
/// terms joined by `_`, raw term counts (no sublinear scaling -- library
/// default), `tf * idf`, L2-normalized row (library default `norm: L2`).
/// Terms not in the vocabulary are ignored, same as `CountVectorizer::transform`.
fn tfidf_transform(
    text: &str,
    vocabulary: &std::collections::HashMap<String, usize>,
    idf: &[f64],
    ngram_range: (usize, usize),
) -> Vec<f32> {
    let tokens: Vec<String> = text.split_whitespace().map(str::to_lowercase).collect();

    let mut tf = vec![0.0_f64; idf.len()];
    for n in ngram_range.0..=ngram_range.1 {
        if n == 0 || tokens.len() < n {
            continue;
        }
        for window in tokens.windows(n) {
            let term = window.join("_");
            if let Some(&idx) = vocabulary.get(&term) {
                tf[idx] += 1.0;
            }
        }
    }

    for (value, &idf_val) in tf.iter_mut().zip(idf.iter()) {
        *value *= idf_val;
    }

    let norm: f64 = tf.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 {
        for value in &mut tf {
            *value /= norm;
        }
    }

    tf.into_iter().map(|v| v as f32).collect()
}
