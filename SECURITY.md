# Security policy

## Supported versions

Only the current private reviewer candidate is maintained. No stable public release exists yet.

## Reporting a vulnerability

Do not open a public issue for credentials, private filenames, path traversal, unsafe archive handling, or confidential data exposure. Email `litaishan@caf.ac.cn` with a minimal description, affected commit, reproduction steps, and impact. Do not include restricted data unless requested through an approved channel.

## Research-use boundary

PruScope is research software, not a safety-critical harvest controller. Predictions require local validation and human oversight. The repository contains no authentication service and should not be exposed as an unauthenticated network endpoint.
