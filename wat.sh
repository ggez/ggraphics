#!/bin/sh
cargo build --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/debug/ggraphics.wasm --out-dir generated --no-modules
rsync -av generated/ icefox@roc.alopex.li:htdocs/temp/g7

