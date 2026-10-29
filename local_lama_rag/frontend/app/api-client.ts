// Thin wrapper around the FastAPI backend (api/main.py, api/schemas.py).
// Types mirror the Pydantic response models 1:1 -- no reinvented contract.

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function createSession(): Promise<string> {
  const res = await fetch(`${API_URL}/sessions`, { method: "POST" });
  const data = await handle<{ session_id: string }>(res);
  return data.session_id;
}

export async function ask(
  sessionId: string,
  question: string,
  targetFiles?: string[],
): Promise<AskResponse> {
  const res = await fetch(`${API_URL}/ask`, {
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

export async function askImage(sessionId: string, question: string, file: File): Promise<AskResponse> {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("question", question);
  form.append("file", file);
  const res = await fetch(`${API_URL}/ask/image`, { method: "POST", body: form });
  return handle<AskResponse>(res);
}

export async function analyzeImage(sessionId: string, file: File): Promise<AskResponse> {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("file", file);
  const res = await fetch(`${API_URL}/ask/vision-analysis`, { method: "POST", body: form });
  return handle<AskResponse>(res);
}

export async function uploadFiles(files: File[]): Promise<UploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${API_URL}/upload`, { method: "POST", body: form });
  return handle<UploadResponse>(res);
}

export async function listFiles(): Promise<FileInfo[]> {
  const res = await fetch(`${API_URL}/files`);
  const data = await handle<{ files: string[]; details: FileInfo[] }>(res);
  return data.details?.length
    ? data.details
    : data.files.map((name) => ({ name, size_bytes: 0, kind: "file" }));
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
