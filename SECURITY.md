# Security policy

Report a vulnerability privately through GitHub's "Report a
vulnerability" button on this repository's Security tab. Please do not
open a public issue for it. You will get a reply within 7 days.

Scope: the code in this repository. level-3-deep/api.py is a study
server: it binds to localhost, has no authentication, TLS or rate
limiting, and is not meant to be exposed to a network. The artifact
loader's hash check detects corruption and tampering, not a rebuilt
manifest and weights pair (level-2-build/README.md, Failure modes).
