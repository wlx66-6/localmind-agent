export type Track = 'cloud' | 'local'

export type Source = {
  id: number
  path: string
  enabled: boolean
  egress_allowed: boolean
  document_count: number
  chunk_count: number
  last_scan_at: string | null
}

export type Citation = {
  index: number
  chunk_id: string
  path: string
  heading: string
  start_line: number
  end_line: number
  score: number
  text: string
  // Position inside the document, and the PDF page range when the source is a
  // PDF (a reflowed page has no stable line numbers to cite).
  ordinal: number
  page: number | null
  page_end: number | null
  // Retrieval trace: per-route ranks, fusion order and rerank score.
  dense_rank: number | null
  sparse_rank: number | null
  fused_rank: number | null
  dense_score: number | null
  rerank_score: number | null
}

export type RetrievalInfo = {
  mode: string
  rerank_failed: boolean
  device: string | null
  timings: {
    embed_ms: number
    dense_ms: number
    sparse_ms: number
    rerank_ms: number
    total_ms: number
  }
}

/** What this turn was about to send, and what the source blacklist withheld. */
export type Egress = {
  track: Track
  enabled: boolean
  chunks: number
  files: number
  chars: number
  history_messages: number
  blocked_chunks: number
  blocked_files: number
  /** Recalled long-term memories injected into the prompt. */
  memories: number
  memory_chars: number
  skipped: boolean
}

export type MemoryKind = 'fact' | 'preference' | 'summary'

/** A memory recalled into this turn (SSE `citations.memories`). */
export type RecalledMemory = {
  id: number
  kind: MemoryKind
  key: string
  content: string
  occurred_at: string | null
  created_at: string
  score: number
}

/** A memory row as listed in the panel; superseded rows keep their history. */
export type MemoryItem = {
  id: number
  kind: MemoryKind
  key: string
  content: string
  occurred_at: string | null
  created_at: string
  status: 'active' | 'superseded'
  superseded_by: number | null
  source_conversation_id: number | null
  embedding_ok: boolean
}

export type MessageMetrics = {
  track?: Track
  model?: string | null
  ttft_ms?: number | null
  total_ms?: number | null
  refused?: boolean
  skipped?: boolean
  blocked?: boolean
  retrieval?: RetrievalInfo
  egress?: Egress
}

export type Health = {
  status: string
  ollama: boolean
  qdrant: boolean
  deepseek_configured: boolean
  local_model: boolean
  default_track: Track
}

export type ScanResult = {
  scanned: number
  indexed: number
  skipped: number
  removed: number
  failed: number
  errors: string[]
}

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  memories: RecalledMemory[]
  cited: number[]
  refused: boolean
  track: Track | null
  egress: Egress | null
  retrieval: RetrievalInfo | null
  ttftMs: number | null
  totalMs: number | null
  skipped: boolean
  blocked: boolean
  error?: string
  streaming: boolean
}