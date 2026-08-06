/**
 * SSE 消费器：POST 请求用 fetch + ReadableStream 解析 `event:/data:` 帧。
 * （EventSource 只支持 GET，无法携带 JSON body，故手写。）
 */

export async function consumeSSE(
  res: Response,
  handlers: Record<string, (data: Record<string, unknown>) => void>,
): Promise<void> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* ignore */
    }
    throw new Error(`HTTP ${res.status}: ${detail}`)
  }
  if (!res.body) throw new Error("响应无 body")
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ""
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const frames = buf.split("\n\n")
    buf = frames.pop() ?? ""
    for (const frame of frames) {
      let event = "message"
      const dataLines: string[] = []
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim()
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim())
      }
      const handler = handlers[event]
      if (handler && dataLines.length) {
        handler(JSON.parse(dataLines.join("\n")) as Record<string, unknown>)
      }
    }
  }
}
