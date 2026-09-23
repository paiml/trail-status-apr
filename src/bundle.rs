//! Bundles the TF-IDF vocabulary + both LogisticRegression classifiers into
//! a single self-contained `.apr` file.
//!
//! Doesn't require any upstream aprender change: `aprender::format::save()`
//! is generic over any `Serialize` type, so this defines its own bundle
//! struct (composed of `LogisticRegression`, which is already `Serialize`
//! in aprender, plus plain vocabulary/IDF data) rather than needing
//! `TfidfVectorizer` itself to be serializable (it isn't -- see
//! ISSUES.md #13). This also means inference no longer needs to re-fit a
//! vectorizer from the training corpus: the vocabulary/IDF travel with the
//! model, and TF-IDF transform at inference time is a small hand-rolled
//! function matching aprender's own formula exactly (raw term counts,
//! `tf * idf`, L2-normalized rows -- see `tfidf_transform` in `model.rs`).

use aprender::classification::LogisticRegression;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
pub struct TrailStatusBundle {
    pub vocabulary: HashMap<String, usize>,
    pub idf: Vec<f64>,
    pub ngram_range: (usize, usize),
    pub blankets_clf: LogisticRegression,
    pub rope_clf: LogisticRegression,
}
