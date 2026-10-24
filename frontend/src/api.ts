import type {
  Citation,
  Egress,
  Health,
  MemoryItem,
  RecalledMemory,
  RetrievalInfo,
  ScanResult,
  Source,
  Track,
} from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    let detail = `请求失败（${response.status}）`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      // 非 JSON 错误体，保留默认信息
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

export const getHealth = () => request<Health>('/api/health')

export const listSources = () => request<Source[]>('/api/sources')

export const addSource = (path: string) =>
  request<Source>('/api/sources', { method: 'POST', body: JSON.stringify({ path }) })

export async function deleteSource(id: number): Promise<void> {
  const response = await fetch(`/api/sources/${id}`, { method: 'DELETE' })
  if (!response.ok) {
    let detail = `删除失败（${response.status}）`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      // 忽略非 JSON 错误体
    }
    throw new Error(detail)
  }
}

export const scanSource = (id: number) =>
  request<ScanResult>(`/api/sources/${id}/scan`, { method: 'POST' })

/** Egress policy per source directory; enforced server-side on the next turn. */
export const updateSourceEgress = (id: number, egressAllowed: boolean) =>
  request<Source>(`/api/sources/${id}`, {
    method: 'PATCH',
    body: JSON.stringify({ egress_allowed: egressAllowed }),
  })

export const listMemories = (includeSuperseded = false) =>
  request<MemoryItem[]>(`/api/memories?include_superseded=${includeSuperseded}`)

export async function deleteMemory(id: number): Promise<void> {
  const response = await fetch(`/api/memories/${id}`, { method: 'DELETE' })
  if (!response.ok) {
    let detail = `删除失败（${response.status}）`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      // 忽略非 JSON 错误体
    }
    throw new Error(detail)
  }
}

export type CitationsPayload = {
  conversationId: number
  track: Track
  items: Citation[]
  memories: RecalledMemory[]
  retrieval: RetrievalInfo
  egress: Egress
  refused: boolean
}

export type DonePayload = {
  conversationId: number
  cited: number[]
  refused: boolean
  track: Track
  skipped: boolean
  blocked: boolean
  ttftMs: number | null
  totalMs: number | null
}

export type ChatHandlers = {
  onCitations: (payload: CitationsPayload) => void
  onToken: (text: string) => void
  onDone: (result: DonePayload) => void
  onError: (message: string) => void
}

type Frame = { event: string; data: unknown }

function parseFrame(frame: string): Frame | null {
  let event = 'message'
  const data: string[] = []
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''))
  }
  if (data.length === 0) return null
  return { event, data: JSON.parse(data.join('\n')) }
}

/** 读取 POST + SSE 的流式响应；EventSource 只支持 GET，所以这里手写解析。 */
export async function streamChat(
  body: {
    message: string
    conversation_id: number | null
    track?: Track
    skip_egress?: boolean
  },
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok) {
    let detail = `请求失败（${response.status}）`
    try {
      const payload = (await response.json()) as { detail?: string }
      if (payload.detail) detail = payload.detail
    } catch {
      // 忽略非 JSON 错误体
    }
    throw new Error(detail)
  }
  if (!response.body) throw new Error('当前浏览器不支持流式响应')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
    let boundary = buffer.indexOf('\n\n')
    while (boundary >= 0) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      const parsed = parseFrame(frame)
      if (parsed) dispatch(parsed, handlers)
      boundary = buffer.indexOf('\n\n')
    }
  }
}

function dispatch(frame: Frame, handlers: ChatHandlers): void {
  switch (frame.event) {
    case 'citations': {
      const data = frame.data as {
        conversation_id: number
        track: Track
        items: Citation[]
        memories?: RecalledMemory[]
        retrieval: RetrievalInfo
        egress: Egress
        refused: boolean
      }
      handlers.onCitations({
        conversationId: data.conversation_id,
        track: data.track,
        items: data.items,
        memories: data.memories ?? [],
        retrieval: data.retrieval,
        egress: data.egress,
        refused: data.refused,
      })
      break
    }
    case 'token':
      handlers.onToken((frame.data as { text: string }).text)
      break
    case 'done': {
      const data = frame.data as {
        conversation_id: number
        cited: number[]
        refused: boolean
        track: Track
        skipped: boolean
        blocked: boolean
        ttft_ms: number | null
        total_ms: number | null
      }
      handlers.onDone({
        conversationId: data.conversation_id,
        cited: data.cited,
        refused: data.refused,
        track: data.track,
        skipped: data.skipped,
        blocked: data.blocked,
        ttftMs: data.ttft_ms,
        totalMs: data.total_ms,
      })
      break
    }
    case 'error':
      handlers.onError((frame.data as { message: string }).message)
      break
    default:
      break
  }
}