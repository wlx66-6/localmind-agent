export type Source = {
  id: number
  path: string
  enabled: boolean
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
  // Retrieval trace: per-route ranks, fusion order and rerank score.
  dense_rank: number | null
  sparse_rank: number | null
  fused_rank: number | null
  dense_score: number | null
  rerank_score: number | null
}

export type Health = {
  status: string
  ollama: boolean
  qdrant: boolean
  deepseek_configured: boolean
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
  cited: number[]
  refused: boolean
  error?: string
  streaming: boolean
}