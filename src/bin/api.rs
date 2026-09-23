//! Trail status inference HTTP API. The model is embedded in this binary
//! at compile time (see `model.rs`) -- no `.apr` file needed at runtime.
//!
//! A minimal, single-threaded example server (not production-hardened --
//! no concurrency, no auth, no rate limiting) demonstrating how to expose
//! the classical trail-status classifiers over HTTP.
//!
//! GET / returns documentation (this same content) as JSON.
//!
//! POST /predict:
//!   Content-Type: application/json
//!   {"text": "blankets creek is open. rope mill remains closed."}
//!
//!   Response:
//!   {"blankets_creek_open":true,"blankets_creek_confidence":0.91,
//!    "rope_mill_open":false,"rope_mill_confidence":0.87}

use clap::Parser;
use tiny_http::{Header, Method, Response, Server};
use trail_status_infer::model::TrailStatusModel;

/// Trail status inference HTTP API (GET / for docs, POST /predict for
/// inference). Use --host 0.0.0.0 in a container -- 127.0.0.1 is only
/// reachable from inside it.
#[derive(Parser)]
#[command(name = "trail-status-api", version, about, long_about = None)]
struct Args {
    /// Address to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on
    #[arg(long, default_value_t = 8080)]
    port: u16,
}

#[derive(serde::Deserialize)]
struct PredictRequest {
    text: String,
}

fn json_header() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap()
}

/// Served on `GET /` -- self-documentation for anyone who hits the root
/// without reading the README first (a browser visit, a healthcheck probe
/// poking around, etc). `bind_addr` is spliced into the curl example so
/// the docs always reflect the address this instance actually bound to.
fn index_body(bind_addr: &str) -> String {
    format!(
        r#"{{
  "service": "trail-status-api",
  "description": "Predicts whether Blankets Creek and Rope Mill (SORBA Woodstock trail systems) are open or closed, given a free-text status update.",
  "endpoints": {{
    "GET /": {{
      "description": "This documentation."
    }},
    "POST /predict": {{
      "description": "Predict open/closed status for both trails from status text.",
      "request_example": {{
        "method": "POST",
        "path": "/predict",
        "headers": {{
          "Content-Type": "application/json"
        }},
        "body": {{
          "text": "blankets creek is open. rope mill remains closed."
        }}
      }},
      "response_example": {{
        "blankets_creek_open": true,
        "blankets_creek_confidence": 0.828,
        "rope_mill_open": false,
        "rope_mill_confidence": 0.940
      }},
      "curl_example": "curl -s -X POST http://{bind_addr}/predict -H 'Content-Type: application/json' -d '{{\"text\": \"blankets creek is open. rope mill remains closed.\"}}'"
    }}
  }}
}}
"#
    )
}

fn main() {
    let args = Args::parse();
    let bind_addr = format!("{}:{}", args.host, args.port);

    println!("Loading embedded model...");
    let model = TrailStatusModel::load();

    let server = Server::http(&bind_addr).unwrap_or_else(|e| panic!("bind {bind_addr}: {e}"));
    println!("Listening on http://{bind_addr}  (GET / for docs, POST /predict for inference)");

    for mut request in server.incoming_requests() {
        let method = request.method().clone();
        let url = request.url().to_string();

        if method == Method::Get && url == "/" {
            let response = Response::from_string(index_body(&bind_addr)).with_header(json_header());
            let _ = request.respond(response);
            continue;
        }

        if method != Method::Post || url != "/predict" {
            let response = Response::from_string("{\"error\":\"GET / for docs, POST /predict for inference\"}")
                .with_status_code(404)
                .with_header(json_header());
            let _ = request.respond(response);
            continue;
        }

        let mut body = String::new();
        if let Err(e) = request.as_reader().read_to_string(&mut body) {
            let response = Response::from_string(format!("{{\"error\":\"failed to read body: {e}\"}}"))
                .with_status_code(400)
                .with_header(json_header());
            let _ = request.respond(response);
            continue;
        }

        let parsed: Result<PredictRequest, _> = serde_json::from_str(&body);
        let response = match parsed {
            Ok(req) => {
                let pred = model.predict(&req.text);
                let json = serde_json::to_string(&pred).expect("serialize prediction");
                Response::from_string(json).with_header(json_header())
            }
            Err(e) => Response::from_string(format!("{{\"error\":\"invalid request: {e}\"}}"))
                .with_status_code(400)
                .with_header(json_header()),
        };
        let _ = request.respond(response);
    }
}
