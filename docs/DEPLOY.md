# Deploying Nexus AI

Nexus is two services: an **API** (FastAPI, needs about 2 GB of RAM because of the vision and
re-ranking models) and a **web app** (Next.js). Pick one of the two setups below.

## Option A: run everything on your own machine or server

```bash
cd local_lama_rag
cp .env.example .env        # set NVIDIA_API_KEY (and a long random AUTH_SECRET on a server)
docker compose up --build
```

Open http://localhost:3000. Accounts, documents and indexes live in the `nexus-users` Docker
volume, so they survive restarts. To put it on a public server, set `CORS_ORIGINS` to your web
address, build the web image with `NEXT_PUBLIC_API_URL` pointing at the API address, and put both
behind HTTPS (Caddy or nginx).

## Option B: free hosted demo (Hugging Face Space for the API + Vercel for the web app)

### 1. Make the API image public
Every release publishes `ghcr.io/karthikps84/nexus_ai_multi-agent_rag-api`. New GHCR packages are
private by default: open the package on GitHub, then **Package settings -> Change visibility ->
Public**.

### 2. Create the API Space
1. On https://huggingface.co/new-space choose **Docker** (blank), hardware **CPU basic**, visibility
   Public.
2. In the Space, add these two files:

   `README.md`
   ```yaml
   ---
   title: Nexus AI API
   sdk: docker
   app_port: 7860
   ---
   ```
   `Dockerfile`
   ```dockerfile
   FROM ghcr.io/karthikps84/nexus_ai_multi-agent_rag-api:latest
   ENV PORT=7860
   ```
3. In **Settings -> Variables and secrets** add:

   | Name | Type | Value |
   |---|---|---|
   | `NVIDIA_API_KEY` | Secret | your NVIDIA key |
   | `AUTH_SECRET` | Secret | a long random string (`python -c "import secrets; print(secrets.token_hex(32))"`) |
   | `CORS_ORIGINS` | Variable | your Vercel URL, e.g. `https://nexus-ai.vercel.app` (add it after step 3) |

The API will be at `https://<your-user>-<space-name>.hf.space`. Check `/health`.

> The free Space has a **temporary disk**: accounts and uploaded documents are wiped whenever it
> restarts. That is fine for a demo. For real use, add persistent storage (paid) and set
> `USERS_ROOT=/data/users`, or use Option A.

### 3. Deploy the web app on Vercel
1. https://vercel.com/new -> import the GitHub repo.
2. **Root Directory:** `local_lama_rag/frontend`. Framework: Next.js (auto-detected).
3. **Environment variable:** `NEXT_PUBLIC_API_URL` = the Space URL from step 2 (no trailing slash).
4. Deploy, then copy the Vercel URL back into the Space's `CORS_ORIGINS` and restart the Space.

### 4. Put the link in the README
Add the live demo link under the badges and the About box of the repo.

## Updating
Releases publish a new `:latest` image. Restart the Space (**Settings -> Factory reboot**) to pull
it; Vercel redeploys on every push to `main` by itself.
