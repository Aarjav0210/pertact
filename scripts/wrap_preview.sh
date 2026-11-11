#!/bin/sh
# Wrap site/index.html in the same minimal skeleton the artifact host adds, so it opens straight from disk.
# site/index.html is a fragment with no doctype or charset; opened as it is, a browser can garble its symbols.
#   sh scripts/wrap_preview.sh              -> site/notebook.html
#   sh scripts/wrap_preview.sh out.html     -> that file
#   sh scripts/wrap_preview.sh some/dir     -> some/dir/index.html
# The page points at figures/ beside it, as published. A copy written into site/ gets ../figures/ so its images load from disk.
case "${1:-site/notebook.html}" in *.html) out="${1:-site/notebook.html}" ;; *) out="$1/index.html" ;; esac
case "$out" in site/*) fix='s|src="figures/|src="../figures/|g' ;; *) fix='' ;; esac
{ printf '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>:root{color-scheme:light;padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom)}body{margin:0;font:14px system-ui}img{max-width:100%%}[hidden]{display:none!important}</style>\n'; sed "$fix" site/index.html; printf '\n</body></html>\n'; } > "$out"
