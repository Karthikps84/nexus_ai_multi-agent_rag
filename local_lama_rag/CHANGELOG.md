# Changelog

## [2.0.0](https://github.com/Karthikps84/nexus_ai_multi-agent_rag/compare/v0.1.1...v2.0.0) (2026-10-29)


### ⚠ BREAKING CHANGES

* every endpoint except /health and /auth/* now requires a bearer token.

### Features

* per-user accounts, streaming answers and document deletion in the API ([f97bbc7](https://github.com/Karthikps84/nexus_ai_multi-agent_rag/commit/f97bbc7ca3304c2a5fe730787e9fe7d4f145be30))
* sign-in screen, streamed answers and delete button in the UI ([51cf7ed](https://github.com/Karthikps84/nexus_ai_multi-agent_rag/commit/51cf7ede94c334c81d69c3ee76334a9848bbe641))


### Bug Fixes

* avoid exposing internal errors in stream SSE response ([8aa78cc](https://github.com/Karthikps84/nexus_ai_multi-agent_rag/commit/8aa78ccda2d6c09628414b0e1acfc02544870e43))
* stop routing 'Docker images' to the image agent and splitting simple 'and' questions ([1f5aee6](https://github.com/Karthikps84/nexus_ai_multi-agent_rag/commit/1f5aee63752effd381c56474dce80842025503ba))


### Documentation

* add docker compose, deployment guide and README updates for v2 ([e25c4ad](https://github.com/Karthikps84/nexus_ai_multi-agent_rag/commit/e25c4adc8576a01d376c39240adbaf3bca735e88))

## [0.1.1](https://github.com/Karthikps84/nexus_ai_multi-agent_rag/compare/v0.1.0...v0.1.1) (2026-10-29)


### Documentation

* add project description ([446adc4](https://github.com/Karthikps84/nexus_ai_multi-agent_rag/commit/446adc4033cbf09c04c3791406f342438f2cb059))
* add project description ([e79eb87](https://github.com/Karthikps84/nexus_ai_multi-agent_rag/commit/e79eb8725c6d837ab48eb8afd3e55e6048c835c9))
