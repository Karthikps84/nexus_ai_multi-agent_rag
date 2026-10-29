// Thin wrapper around the FastAPI backend (api/main.py, api/schemas.py).
// Types mirror the Pydantic response models 1:1 -- no reinvented contract.

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const TOKEN_KEY = "nexus_token";

export interface RetrievedDoc {
  content: string;
  metadata: Record<string, unknown>;
}

export interface AskResponse {
  answer: string;
  sources: string[];
  route: string;
  context: string;
  dataframe: unknown;
  verification: { confidence?: number; unsupported_claims?: unknown } | null;
  max_score: number;
  docs: RetrievedDoc[];
  execution_plan: Record<string, unknown>;
  citations: string[];
  confidence: number | null;
  execution_time: number;
  cache_hits: Record<string, boolean>;
  token_usage: Record<string, number>;
  trace: unknown[];
  annotated_image?: string | null;
}

export interface FileInfo {
  name: string;
  size_bytes: number;
  kind: string;
}

export interface UploadResponse {
  saved: string[];
  skipped: string[];
  files_indexed: number;
}

export class AuthExpiredError extends Error {}

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

function setToken(token: string) {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* storage unavailable: user will have to sign in again next visit */
  }
}

export function clearToken() {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore */
  }
}

async function errorMessage(res: Response): Promise<string> {
  const body = await res.text().catch(() => "");
  try {
    const detail = JSON.parse(body).detail;
    if (typeof detail === "string") return detail;
  } catch {
    /* not JSON */
  }
  return `${res.status} ${res.statusText}${body ? `: ${body}` : ""}`;
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json() as Promise<T>;
}

async function authFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(url, { ...init, headers });
  if (res.status === 401) {
    clearToken();
    throw new AuthExpiredError("Your session expired. Please sign in again.");
  }
  return res;
}

// ── accounts ────────────────────────────────────────────────────────────────
async function authRequest(path: string, username: string, password: string): Promise<string> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  const data = await handle<{ token: string; username: string }>(res);
  setToken(data.token);
  return data.username;
}

export const login = (username: string, password: string) => authRequest("/auth/login", username, password);
export const register = (username: string, password: string) => authRequest("/auth/register", username, password);

export async function fetchMe(): Promise<string> {
  const res = await authFetch(`${API_URL}/auth/me`);
  return (await handle<{ username: string }>(res)).username;
}

// ── chat ────────────────────────────────────────────────────────────────────
export async function createSession(): Promise<string> {
  const res = await authFetch(`${API_URL}/sessions`, { method: "POST" });
  const data = await handle<{ session_id: string }>(res);
  return data.session_id;
}

export async function ask(
  sessionId: string,
  question: string,
  targetFiles?: string[],
): Promise<AskResponse> {
  const res = await authFetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      question,
      target_files: targetFiles && targetFiles.length ? targetFiles : null,
    }),
  });
  return handle<AskResponse>(res);
}

export interface StreamHandlers {
  onMeta?: (meta: { route: string; sources: string[] }) => void;
  onToken: (text: string) => void;
  onDone: (result: AskResponse) => void;
}

// Reads the /ask/stream server-sent events: meta, token*, done (or error).
export async function askStream(
  sessionId: string,
  question: string,
  targetFiles: string[] | undefined,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await authFetch(`${API_URL}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      question,
      target_files: targetFiles && targetFiles.length ? targetFiles : null,
    }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(await errorMessage(res));

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (block: string) => {
    let event = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7);
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    if (!data) return;
    const payload = JSON.parse(data);
    if (event === "meta") handlers.onMeta?.(payload);
    else if (event === "token") handlers.onToken(payload.t);
    else if (event === "done") handlers.onDone(payload as AskResponse);
    else if (event === "error") throw new Error(payload.detail ?? "Stream failed.");
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      dispatch(block);
    }
  }
}

export async function askImage(sessionId: string, question: string, file: File): Promise<AskResponse> {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("question", question);
  form.append("file", file);
  const res = await authFetch(`${API_URL}/ask/image`, { method: "POST", body: form });
  return handle<AskResponse>(res);
}

export async function analyzeImage(sessionId: string, file: File): Promise<AskResponse> {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("file", file);
  const res = await authFetch(`${API_URL}/ask/vision-analysis`, { method: "POST", body: form });
  return handle<AskResponse>(res);
}

// ── documents ───────────────────────────────────────────────────────────────
export async function uploadFiles(files: File[]): Promise<UploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await authFetch(`${API_URL}/upload`, { method: "POST", body: form });
  return handle<UploadResponse>(res);
}

export async function listFiles(): Promise<FileInfo[]> {
  const res = await authFetch(`${API_URL}/files`);
  const data = await handle<{ files: string[]; details: FileInfo[] }>(res);
  return data.details?.length
    ? data.details
    : data.files.map((name) => ({ name, size_bytes: 0, kind: "file" }));
}

export async function deleteFile(name: string): Promise<string[]> {
  const res = await authFetch(`${API_URL}/files/${encodeURIComponent(name)}`, { method: "DELETE" });
  return (await handle<{ files: string[] }>(res)).files;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`);
    if (!res.ok) return false;
    const data = await handle<{ status: string; initialized: boolean }>(res);
    return data.status === "ok" && data.initialized;
  } catch {
    return false;
  }
}
