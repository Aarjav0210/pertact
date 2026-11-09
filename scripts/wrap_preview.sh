#!/bin/sh
# Wrap site/index.html in the same minimal skeleton the artifact host adds, for local testing.
out="$1/index.html"
{ printf '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>:root{color-scheme:light;padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom)}body{margin:0;font:14px system-ui}img{max-width:100%%}[hidden]{display:none!important}</style>\n'; cat site/index.html; printf '\n</body></html>\n'; } > "$out"
