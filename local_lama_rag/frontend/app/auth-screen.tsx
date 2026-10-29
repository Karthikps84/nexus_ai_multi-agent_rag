"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { login, register } from "./api-client";

export default function AuthScreen({ onAuthed }: { onAuthed: (username: string) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const name = await (mode === "login" ? login : register)(username, password);
      onAuthed(name);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  const input =
    "w-full rounded-xl border border-zinc-300 bg-[var(--background)] px-4 py-3 text-sm text-[var(--foreground)] outline-none transition-colors focus:border-indigo-500 dark:border-zinc-700";

  return (
    <div className="flex flex-1 items-center justify-center px-6 py-12">
      <motion.form
        onSubmit={submit}
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-sm rounded-2xl border border-[var(--surface-border)] bg-[var(--surface)] p-8 shadow-xl"
      >
        <div className="mb-6 text-center">
          <div className="gradient-ring mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-2xl text-lg font-bold text-white shadow-lg shadow-indigo-500/30">
            N
          </div>
          <h1 className="gradient-text text-2xl font-bold">Nexus AI</h1>
          <p className="mt-1 text-sm text-zinc-400">
            {mode === "login" ? "Sign in to your documents" : "Create your private library"}
          </p>
        </div>

        <div className="flex flex-col gap-3">
          <input
            className={input}
            placeholder="Username"
            autoComplete="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
          <input
            className={input}
            type="password"
            placeholder={mode === "register" ? "Password (8+ characters)" : "Password"}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>

        {error && (
          <p className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700 dark:bg-red-950 dark:text-red-300">
            {error}
          </p>
        )}

        <motion.button
          whileTap={{ scale: 0.97 }}
          type="submit"
          disabled={busy}
          className="gradient-ring mt-5 w-full rounded-xl py-3 text-sm font-medium text-white shadow-md shadow-indigo-500/20 disabled:opacity-50"
        >
          {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
        </motion.button>

        <button
          type="button"
          onClick={() => {
            setMode(mode === "login" ? "register" : "login");
            setError(null);
          }}
          className="mt-4 w-full text-center text-xs text-zinc-500 hover:text-indigo-500"
        >
          {mode === "login" ? "New here? Create an account" : "Already have an account? Sign in"}
        </button>
        <p className="mt-4 text-center text-[11px] text-zinc-400">
          Each account gets its own documents and index.
        </p>
      </motion.form>
    </div>
  );
}
