# Pinned KataGo rule oracle

The reference commit is exactly:

f6bc4b19a1686caa2d088b56251e8c11c8be6d51

ensure_pinned_source.sh stores a detached checkout under
.cache/katago-reference/<sha>/ and refuses to return a different commit.
No system package installation is performed by this harness.

The oracle protocol is JSON Lines: one request produces one response. Its
adapter owns JSON and coordinate conversion only; legality, captures, ko,
phase transitions, and scoring must come from KataGo Board and BoardHistory.

Build locally with tools/katago_reference/build_oracle.sh.
