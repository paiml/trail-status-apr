# Containerizes the trail-status HTTP API (src/bin/api.rs).
#
# IMPORTANT -- build context must be the PARENT of both this repo and
# aprender, not this directory alone. This project's Cargo.toml depends
# on aprender via a local path dependency (aprender isn't published to
# crates.io), so the aprender source tree has to be present in the build
# context for `cargo build` to see it. Build with:
#
#   cd ~/code
#   docker build -f trail-status-apr/Dockerfile -t trail-status-api .
#
# Run with (0.0.0.0 bind is required -- 127.0.0.1 inside a container is
# only reachable from inside it):
#
#   docker run --rm -p 8080:8080 trail-status-api
#
# Then:
#   curl http://localhost:8080/
#   curl -s -X POST http://localhost:8080/predict \
#     -H 'Content-Type: application/json' \
#     -d '{"text": "blankets creek is open. rope mill remains closed."}'

# syntax=docker/dockerfile:1

FROM rust:1.93-bookworm AS builder

# Matches the absolute path aprender = { path = "/home/alfredo/code/aprender", ... }
# in Cargo.toml, so no Cargo.toml edits are needed to build in-container.
WORKDIR /home/alfredo/code
# --exclude=target skips each repo's (huge -- aprender's alone is 60GB+)
# local build cache, which is useless in-container anyway since we
# recompile fresh here. Avoids needing a .dockerignore outside this repo.
COPY --exclude=target aprender ./aprender
COPY --exclude=target trail-status-apr ./trail-status-apr

WORKDIR /home/alfredo/code/trail-status-apr
RUN cargo build --release --bin trail-status-api

# The model is embedded into the trail-status-api binary at compile time
# (src/model.rs, include_bytes! + aprender::format::load_from_bytes), so
# the runtime stage only needs the binary itself -- no models/ directory,
# no .apr file, nothing else copied in.
FROM debian:bookworm-slim AS runtime
WORKDIR /app

COPY --from=builder /home/alfredo/code/trail-status-apr/target/release/trail-status-api ./trail-status-api

EXPOSE 8080
ENTRYPOINT ["./trail-status-api"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
