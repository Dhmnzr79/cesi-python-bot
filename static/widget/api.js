/**
 * Слой HTTP к /ask (без UI).
 * @param {string} apiBase — пустая строка = тот же origin
 * @param {Record<string, unknown>} body
 */
export async function postAsk(apiBase, body) {
  const base = (apiBase || "").replace(/\/$/, "");
  const url = `${base}/ask`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = {};
  try {
    data = await res.json();
  } catch {
    data = {};
  }
  if (!res.ok) {
    const err = typeof data.error === "string" ? data.error : res.statusText;
    throw new Error(err || "request_failed");
  }
  return data;
}

/**
 * Стриминговый вызов /ask/stream через SSE (fetch + ReadableStream).
 *
 * Протокол:
 *   event: text_delta  data: {"delta": "..."}   — токен ответа
 *   event: ui          data: {полный payload}    — UI после генерации
 *   event: done        data: {}                  — конец стрима
 *
 * @param {string} apiBase
 * @param {Record<string, unknown>} body
 * @param {{
 *   onDelta?: (delta: string) => void,
 *   onUi?: (data: unknown) => void,
 *   onDone?: () => void,
 *   onError?: (msg: string) => void,
 * }} callbacks
 */
export async function streamAsk(apiBase, body, { onDelta, onUi, onDone, onError } = {}) {
  const base = (apiBase || "").replace(/\/$/, "");
  const url = `${base}/ask/stream`;
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      let errMsg = res.statusText || "request_failed";
      try {
        const d = await res.json();
        if (typeof d.error === "string") errMsg = d.error;
      } catch { /* ignore */ }
      throw new Error(errMsg);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let currentEvent = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE: разбиваем по \n, неполную последнюю строку оставляем в буфере
      const parts = buffer.split("\n");
      buffer = parts.pop() ?? "";

      for (const line of parts) {
        if (line.startsWith("event: ")) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
            if (currentEvent === "text_delta") onDelta?.(String(data.delta ?? ""));
            else if (currentEvent === "ui") onUi?.(data);
            else if (currentEvent === "done") onDone?.();
          } catch { /* ignore malformed SSE data */ }
          currentEvent = "";
        }
      }
    }
  } catch (e) {
    onError?.(e instanceof Error ? e.message : "Ошибка сети");
  }
}
