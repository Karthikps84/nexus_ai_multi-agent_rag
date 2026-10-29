"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  analyzeImage,
  ask,
  askImage,
  checkHealth,
  createSession,
  listFiles,
  uploadFiles,
  type AskResponse,
  type FileInfo,
  type RetrievedDoc,
} from "./api-client";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources?: string[];
  confidence?: number | null;
  route?: string;
  latency?: number;
  docs?: RetrievedDoc[];
  image?: string;
  annotated?: string | null;
}

const SESSION_KEY = "local_lama_rag_session_id";

function linkifyCitations(text: string): string {
  return text.replace(/\[([^\[\]]+?\.(?:pdf|docx|xlsx|xls|csv|pptx|txt|md)(?:\s+p\.\d+)?)\]/gi, (_m, label: string) => {
    const short = label.length > 28 ? label.slice(0, 25) + "… " + (label.match(/p\.\d+/)?.[0] ?? "") : label;
    return `[${short}](cite:${encodeURIComponent(label)})`;
  });
}

function formatSize(bytes: number): string {
  if (!bytes) return "";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function DocsPanel({
  files,
  loading,
  fresh,
  scope,
  onToggle,
  onRefresh,
  onClose,
}: {
  files: FileInfo[];
  loading: boolean;
  fresh: Set<string>;
  scope: string[];
  onToggle: (name: string) => void;
  onRefresh: () => void;
  onClose: () => void;
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center justify-between px-4 py-4">
        <div>
          <h2 className="text-sm font-semibold text-[var(--foreground)]">Indexed documents</h2>
          <p className="text-[11px] text-zinc-400 dark:text-zinc-500">
            {files.length} file{files.length === 1 ? "" : "s"} · click to search only that file
          </p>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={onRefresh}
            title="Refresh"
            className="rounded-lg p-1.5 text-zinc-400 transition-colors hover:bg-indigo-500/10 hover:text-indigo-500"
          >
            <motion.svg
              width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
              animate={loading ? { rotate: 360 } : { rotate: 0 }}
              transition={loading ? { duration: 0.9, repeat: Infinity, ease: "linear" } : {}}
            >
              <path d="M21 12a9 9 0 1 1-3-6.7M21 4v5h-5" />
            </motion.svg>
          </button>
          <button
            onClick={onClose}
            title="Close"
            className="rounded-lg p-1.5 text-zinc-400 hover:bg-indigo-500/10 hover:text-indigo-500 "
          >
            ✕
          </button>
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-4">
        {files.length === 0 && !loading && (
          <p className="px-2 py-6 text-center text-xs text-zinc-400">No documents yet. Upload one with +</p>
        )}
        <AnimatePresence initial={false}>
          {files.map((f) => {
            const active = scope.includes(f.name);
            return (
              <motion.button
                key={f.name}
                layout
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                onClick={() => onToggle(f.name)}
                className={`mb-1.5 flex w-full items-center gap-3 rounded-xl border px-3 py-2 text-left transition-colors ${
                  active
                    ? "border-indigo-500 bg-indigo-500/10"
                    : "border-transparent hover:border-[var(--surface-border)] hover:bg-indigo-500/5"
                }`}
              >
                <span className="gradient-ring flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-[9px] font-bold uppercase text-white">
                  {f.kind.slice(0, 4)}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-xs font-medium text-[var(--foreground)]" title={f.name}>
                    {f.name}
                  </span>
                  <span className="block text-[10px] text-zinc-400">{formatSize(f.size_bytes)}</span>
                </span>
                {fresh.has(f.name) && (
                  <span className="rounded-full bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-500">
                    new
                  </span>
                )}
              </motion.button>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-full bg-indigo-500/10 px-2 py-0.5 text-[11px] font-medium text-indigo-500 dark:text-indigo-300">
      {children}
    </span>
  );
}

function SourcesDisclosure({ docs }: { docs: RetrievedDoc[] }) {
  const [open, setOpen] = useState(false);
  if (docs.length === 0) return null;

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen((v) => !v)}
        className="text-xs font-medium text-indigo-500 hover:underline dark:text-indigo-400"
      >
        {open ? "Hide" : "Show"} sources ({docs.length})
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-2 flex flex-col gap-2">
              {docs.map((d, i) => {
                const file = (d.metadata?.source_file as string) ?? "unknown";
                const page = d.metadata?.page;
                return (
                  <div
                    key={i}
                    className="rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-xs text-zinc-600 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
                  >
                    <div className="mb-1 font-medium text-zinc-500 dark:text-zinc-400">
                      {file}
                      {page != null ? ` · p.${page}` : ""}
                    </div>
                    <p className="line-clamp-3">{d.content}</p>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 px-1 py-1">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="h-2 w-2 rounded-full bg-indigo-400"
          animate={{ y: [0, -5, 0] }}
          transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.15 }}
        />
      ))}
    </div>
  );
}

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<FileInfo[]>([]);
  const filesRef = useRef<FileInfo[]>([]);
  useEffect(() => {
    filesRef.current = files;
  }, [files]);
  const [filesLoading, setFilesLoading] = useState(false);
  const [fresh, setFresh] = useState<Set<string>>(new Set());
  const [scope, setScope] = useState<string[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageMode, setImageMode] = useState<"ask" | "detect">("detect");

  async function refreshFiles(): Promise<FileInfo[]> {
    setFilesLoading(true);
    try {
      const list = await listFiles();
      setFiles(list);
      return list;
    } catch {
      return [];
    } finally {
      setFilesLoading(false);
    }
  }

  function toggleScope(name: string) {
    setScope((prev) => (prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name]));
  }

  function handlePickImage(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f) return;
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImageFile(f);
    setImagePreview(URL.createObjectURL(f));
  }

  function clearImage() {
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImageFile(null);
    setImagePreview(null);
  }

  useEffect(() => {
    checkHealth().then(setApiOnline);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial data fetch on mount
    refreshFiles();
  }, []);

  useEffect(() => {
    const stored = sessionStorage.getItem(SESSION_KEY);
    if (stored) {
      // sessionStorage is client-only; reading it in render would break SSR hydration
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSessionId(stored);
      return;
    }
    let cancelled = false;
    createSession()
      .then((id) => {
        if (cancelled) return;
        sessionStorage.setItem(SESSION_KEY, id);
        setSessionId(id);
      })
      .catch(() => {
        if (!cancelled) setError("Could not reach the API. Is uvicorn running on port 8000?");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isSending]);

  async function handleSend() {
    const question = input.trim();
    const hasImage = imageFile !== null;
    if ((!question && !hasImage) || !sessionId || isSending) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text: question || (imageMode === "detect" ? "Detect objects in this image" : "Describe this image"),
      image: imagePreview ?? undefined,
    };
    const file = imageFile;
    const mode = imageMode;
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setImageFile(null);
    setImagePreview(null);
    setIsSending(true);
    setError(null);

    try {
      let result: AskResponse;
      if (file && mode === "detect") result = await analyzeImage(sessionId, file);
      else if (file) result = await askImage(sessionId, question, file);
      else result = await ask(sessionId, question, scope);
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: result.answer,
          sources: result.sources,
          confidence: result.confidence,
          route: result.route,
          latency: result.execution_time,
          docs: result.docs,
          annotated: result.annotated_image,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
    } finally {
      setIsSending(false);
    }
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = Array.from(e.target.files ?? []);
    e.target.value = "";
    if (picked.length === 0) return;

    setIsUploading(true);
    setError(null);
    setNotice(null);
    try {
      const before = new Set(filesRef.current.map((f) => f.name));
      const result = await uploadFiles(picked);
      const list = await refreshFiles();
      const added = new Set(list.map((f) => f.name).filter((n) => !before.has(n) || result.saved.includes(n)));
      setFresh(added);
      setSidebarOpen(true);
      setTimeout(() => setFresh(new Set()), 8000);
      const parts: string[] = [];
      if (result.saved.length) parts.push(`indexed: ${result.saved.join(", ")}`);
      if (result.skipped.length) parts.push(`skipped (unsupported): ${result.skipped.join(", ")}`);
      setNotice(parts.join(" — ") || "No files uploaded.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const suggestions = [
    "Summarize the uploaded documents",
    "Compare the key documents",
    "What's in the data folder?",
  ];

  return (
    <div className="flex flex-1 bg-[var(--background)]">
      <div className="flex min-w-0 flex-1 flex-col">
      <header className="flex items-center justify-between border-b border-zinc-200/70 bg-[var(--surface)]/80 px-6 py-4 backdrop-blur-sm dark:border-zinc-800/70">
        <div className="flex items-center gap-3">
          <div className="gradient-ring flex h-9 w-9 items-center justify-center rounded-xl text-white shadow-md shadow-indigo-500/20">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2a7 7 0 0 0-7 7v3.5L3 17h4a5 5 0 0 0 10 0h4l-2-4.5V9a7 7 0 0 0-7-7Z" />
            </svg>
          </div>
          <div>
            <h1 className="gradient-text text-lg font-bold leading-tight">RAG Assistant</h1>
            <p className="text-[11px] text-zinc-400 dark:text-zinc-500">ask your documents anything</p>
          </div>
        </div>
        <div className="flex items-center gap-3 text-sm text-zinc-500 dark:text-zinc-400">
          <div className="relative">
            <button
              onClick={() => setSidebarOpen((v) => !v)}
              aria-expanded={sidebarOpen}
              className={`flex items-center gap-2 rounded-xl border px-3 py-1.5 text-xs font-medium transition-colors ${
                sidebarOpen
                  ? "border-indigo-500 bg-indigo-500/10 text-indigo-500"
                  : "border-[var(--surface-border)] hover:border-indigo-500 hover:text-indigo-500"
              }`}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" /><path d="M14 3v6h6" />
              </svg>
              Indexed docs
              <span className="rounded-full bg-indigo-500/15 px-1.5 text-[10px] font-semibold text-indigo-500">
                {files.length}
              </span>
              <motion.span animate={{ rotate: sidebarOpen ? 180 : 0 }} className="text-[10px]">
                ▾
              </motion.span>
            </button>
            <AnimatePresence>
              {sidebarOpen && (
                <>
                  <div className="fixed inset-0 z-20" onClick={() => setSidebarOpen(false)} />
                  <motion.div
                    initial={{ opacity: 0, y: -6, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -6, scale: 0.98 }}
                    transition={{ duration: 0.15 }}
                    className="absolute right-0 top-full z-30 mt-2 flex max-h-[65vh] w-[min(20rem,calc(100vw-2rem))] flex-col overflow-hidden rounded-2xl border border-[var(--surface-border)] bg-[var(--surface)] text-left shadow-2xl"
                  >
                    <DocsPanel
                      files={files}
                      loading={filesLoading}
                      fresh={fresh}
                      scope={scope}
                      onToggle={toggleScope}
                      onRefresh={refreshFiles}
                      onClose={() => setSidebarOpen(false)}
                    />
                  </motion.div>
                </>
              )}
            </AnimatePresence>
          </div>
          <motion.span
            className={`h-2 w-2 rounded-full ${
              apiOnline === null ? "bg-zinc-400" : apiOnline ? "bg-emerald-500" : "bg-red-500"
            }`}
            animate={apiOnline ? { scale: [1, 1.3, 1] } : {}}
            transition={{ duration: 1.6, repeat: Infinity }}
          />
          {apiOnline === null ? "checking…" : apiOnline ? "online" : "offline"}
        </div>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto flex max-w-4xl flex-col gap-5">
          {messages.length === 0 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-12 flex flex-col items-center gap-5 text-center"
            >
              <div className="gradient-ring flex h-14 w-14 items-center justify-center rounded-2xl text-white shadow-lg shadow-indigo-500/30">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 2a7 7 0 0 0-7 7v3.5L3 17h4a5 5 0 0 0 10 0h4l-2-4.5V9a7 7 0 0 0-7-7Z" />
                </svg>
              </div>
              <div>
                <p className="text-base font-semibold text-zinc-700 dark:text-zinc-200">
                  Ask anything about your documents
                </p>
                <p className="mt-1 text-sm text-zinc-400 dark:text-zinc-500">
                  Upload a file below, or just start typing.
                </p>
              </div>
              <div className="flex flex-wrap justify-center gap-2">
                {suggestions.map((s) => (
                  <button
                    key={s}
                    onClick={() => setInput(s)}
                    className="rounded-full border border-zinc-200 bg-[var(--surface)] px-3 py-1.5 text-xs text-zinc-500 transition-colors hover:border-indigo-400 hover:text-indigo-500 dark:border-zinc-700 dark:text-zinc-400"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </motion.div>
          )}
          <AnimatePresence initial={false}>
            {messages.map((m) => (
              <motion.div
                key={m.id}
                initial={{ opacity: 0, y: 12, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.25, ease: "easeOut" }}
                className={`flex gap-2 ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {m.role === "assistant" && (
                  <div className="gradient-ring mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[11px] font-bold text-white">
                    AI
                  </div>
                )}
                <div
                  className={`min-w-0 rounded-2xl px-4 py-3 text-sm shadow-sm sm:px-5 ${
                    m.role === "assistant" ? "max-w-[92%] sm:max-w-[88%]" : "max-w-[85%] sm:max-w-[70%]"
                  } ${
                    m.role === "user"
                      ? "gradient-ring text-white"
                      : "border border-[var(--surface-border)] bg-[var(--surface)] text-[var(--foreground)]"
                  }`}
                >
                  {m.role === "assistant" ? (
                    <div className="msg-markdown">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        urlTransform={(url) => url}
                        components={{
                          a: ({ href, children }) =>
                            href?.startsWith("cite:") ? (
                              <span
                                title={decodeURIComponent(href.slice(5))}
                                className="mx-0.5 inline-flex items-center rounded-md bg-indigo-500/10 px-1.5 py-0.5 align-baseline text-[11px] font-medium text-indigo-500 dark:text-indigo-300"
                              >
                                {children}
                              </span>
                            ) : (
                              <a href={href} target="_blank" rel="noreferrer">
                                {children}
                              </a>
                            ),
                        }}
                      >
                        {linkifyCitations(m.text)}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <>
                      {m.image && (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img src={m.image} alt="attached" className="mb-2 max-h-56 rounded-lg" />
                      )}
                      <p className="whitespace-pre-wrap">{m.text}</p>
                    </>
                  )}
                  {m.annotated && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={m.annotated}
                      alt="detections"
                      className="mt-3 max-h-96 w-auto rounded-lg border border-[var(--surface-border)]"
                    />
                  )}
                  {m.role === "assistant" &&
                    (m.sources?.length || m.confidence != null || m.latency != null) && (
                      <div className="mt-2 flex flex-wrap items-center gap-1.5 border-t border-[var(--surface-border)] pt-2">
                        {m.confidence != null && <Badge>confidence {m.confidence.toFixed(2)}</Badge>}
                        {m.route && <Badge>{m.route}</Badge>}
                        {m.latency != null && <Badge>{m.latency.toFixed(1)}s</Badge>}
                        {m.docs && m.docs.length > 0 && <SourcesDisclosure docs={m.docs} />}
                      </div>
                    )}
                </div>
                {m.role === "user" && (
                  <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-zinc-300 text-[11px] font-bold text-zinc-700 dark:bg-zinc-700 dark:text-zinc-200">
                    You
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
          {isSending && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-2"
            >
              <div className="gradient-ring mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[11px] font-bold text-white">
                AI
              </div>
              <div className="rounded-2xl border border-[var(--surface-border)] bg-[var(--surface)] px-4 py-2 shadow-sm">
                <TypingIndicator />
              </div>
            </motion.div>
          )}
          {notice && (
            <div className="rounded-lg bg-emerald-50 px-4 py-2 text-sm text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300">
              {notice}
            </div>
          )}
          {error && (
            <div className="rounded-lg bg-red-50 px-4 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">
              {error}
            </div>
          )}
        </div>
      </div>

      <div className="border-t border-zinc-200/70 bg-[var(--surface)]/90 px-6 py-4 backdrop-blur-sm dark:border-zinc-800/70">
        <div className="mx-auto max-w-3xl">
          {scope.length > 0 && (
            <div className="mb-2 flex flex-wrap items-center gap-1.5 text-[11px] text-zinc-500">
              <span>Searching only in:</span>
              {scope.map((n) => (
                <button
                  key={n}
                  onClick={() => toggleScope(n)}
                  className="inline-flex max-w-[200px] items-center gap-1 rounded-full bg-indigo-500/10 px-2 py-0.5 font-medium text-indigo-500"
                >
                  <span className="truncate">{n}</span> ✕
                </button>
              ))}
            </div>
          )}
          {imagePreview && (
            <div className="mb-2 flex items-center gap-3 rounded-xl border border-[var(--surface-border)] bg-[var(--background)] p-2">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={imagePreview} alt="to analyze" className="h-14 w-14 rounded-lg object-cover" />
              <div className="flex flex-1 flex-wrap gap-1.5 text-xs">
                {(["detect", "ask"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setImageMode(m)}
                    className={`rounded-full px-3 py-1 font-medium transition-colors ${
                      imageMode === m
                        ? "gradient-ring text-white"
                        : "border border-[var(--surface-border)] text-zinc-500 hover:text-indigo-500"
                    }`}
                  >
                    {m === "detect" ? "Detect objects" : "Ask / classify"}
                  </button>
                ))}
                <span className="self-center text-[11px] text-zinc-400">
                  {imageMode === "detect" ? "YOLO boxes + description" : "type a question, e.g. what is this?"}
                </span>
              </div>
              <button onClick={clearImage} className="px-2 text-zinc-400 hover:text-red-500">
                ✕
              </button>
            </div>
          )}
        <div className="flex items-end gap-3">
          <input
            ref={imageInputRef}
            type="file"
            accept="image/*"
            onChange={handlePickImage}
            className="hidden"
          />
          <button
            onClick={() => imageInputRef.current?.click()}
            title="Attach an image for vision tasks"
            className="rounded-xl border border-zinc-300 px-3 py-3 text-sm text-zinc-500 transition-colors hover:border-indigo-500 hover:text-indigo-500 dark:border-zinc-700 dark:text-zinc-400"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="m21 15-5-5L5 21" />
            </svg>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleUpload}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            title="Upload documents (they get indexed)"
            className="rounded-xl border border-zinc-300 px-3 py-3 text-sm text-zinc-500 transition-colors hover:border-indigo-500 hover:text-indigo-500 disabled:opacity-40 dark:border-zinc-700 dark:text-zinc-400"
          >
            {isUploading ? "…" : "+"}
          </button>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question…"
            rows={1}
            disabled={!sessionId || isSending}
            className="flex-1 resize-none rounded-xl border border-zinc-300 bg-[var(--background)] px-4 py-3 text-sm text-[var(--foreground)] outline-none transition-colors focus:border-indigo-500 disabled:opacity-50 dark:border-zinc-700"
          />
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={handleSend}
            disabled={!sessionId || isSending || (!input.trim() && !imageFile)}
            className="gradient-ring rounded-xl px-5 py-3 text-sm font-medium text-white shadow-md shadow-indigo-500/20 transition-opacity disabled:opacity-40"
          >
            Send
          </motion.button>
        </div>
        </div>
      </div>
      </div>
    </div>
  );
}
