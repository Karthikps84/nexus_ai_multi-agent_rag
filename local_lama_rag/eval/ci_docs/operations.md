# Nexus AI operations

Indexing is incremental. The index remembers each file's size and modified time, and on every
change only new or changed files are embedded, while the chunks of removed files are deleted.
Deleting a document in the UI removes its chunks from the index and the file from disk.

The API streams answers to the browser with server-sent events (SSE): a meta event, many token
events, and a final done event. Accounts use scrypt password hashing and signed tokens, and every
user has a separate document folder and a separate index, so one user never sees another's files.

Every pull request runs continuous integration: ruff linting, pytest tests, and a frontend lint
and build. Versions are produced by release-please from Conventional Commit messages, which opens
a release pull request, writes the changelog, and creates the git tag. Docker images for the API
and the web app are published to GHCR (the GitHub Container Registry) on every release.
