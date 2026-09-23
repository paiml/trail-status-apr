//! Trail status inference CLI. The model is embedded in this binary at
//! compile time (see `model.rs`) -- no `.apr` file needed at runtime.

use clap::Parser;
use trail_status_infer::model::TrailStatusModel;

/// Predict whether Blankets Creek and Rope Mill are open or closed from a
/// free-text trail status update.
#[derive(Parser)]
#[command(name = "trail-status-cli", version, about, long_about = None)]
struct Args {
    /// Trail status text to classify, e.g. "blankets creek is open. rope
    /// mill remains closed."
    text: String,
}

fn main() {
    let args = Args::parse();

    let model = TrailStatusModel::load();
    let pred = model.predict(&args.text);

    println!("input: {:?}", args.text);
    println!(
        "  Blankets Creek: {} ({:.1}% confidence)",
        if pred.blankets_creek_open { "OPEN" } else { "CLOSED" },
        pred.blankets_creek_confidence * 100.0
    );
    println!(
        "  Rope Mill:      {} ({:.1}% confidence)",
        if pred.rope_mill_open { "OPEN" } else { "CLOSED" },
        pred.rope_mill_confidence * 100.0
    );
}
