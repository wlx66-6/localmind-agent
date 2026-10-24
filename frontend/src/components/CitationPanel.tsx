import { localDate } from '../format'
import type { ChatMessage, Citation, Egress, RecalledMemory } from '../types'

function traceLabel(citation: Citation): string {
  const parts: string[] = []
  if (citation.dense_rank !== null) parts.push(`稠密 #${citation.dense_rank}`)
  if (citation.sparse_rank !== null) parts.push(`稀疏 #${citation.sparse_rank}`)
  if (citation.fused_rank !== null) parts.push(`融合 #${citation.fused_rank}`)
  if (citation.rerank_score !== null) {
    parts.push(`重排 ${citation.rerank_score.toFixed(3)}`)
  } else if (citation.dense_score !== null) {
    parts.push(`相似度 ${citation.dense_score.toFixed(3)}`)
  }
  return parts.join(' · ')
}

const KIND_LABEL: Record<RecalledMemory['kind'], string> = {
  fact: '事实',
  preference: '偏好',
  summary: '摘要',
}

function memoryDate(memory: RecalledMemory): string {
  const occurred = memory.occurred_at ? `事发 ${memory.occurred_at}` : null
  const recorded = `记于 ${localDate(memory.created_at)}`
  return occurred ? `${occurred} · ${recorded}` : recorded
}

type Receipt = { title: string; detail: string; tone: string }

/** Turns the server-provided manifest into an honest one-line receipt. */
function receipt(egress: Egress): Receipt {
  const memoryPart =
    egress.memories > 0 ? `，含 ${egress.memories} 条记忆（${egress.memory_chars} 字符）` : ''
  if (egress.track === 'local') {
    return {
      title: '本地生成 · 零出站',
      detail: `本机处理 ${egress.chunks} 段 / ${egress.files} 个文件${memoryPart}，未向任何外部服务发送内容`,
      tone: 'border-emerald-200 bg-emerald-50 text-emerald-800',
    }
  }
  if (egress.skipped) {
    return {
      title: '本次未上传',
      detail: `检索到 ${egress.chunks} 段${memoryPart}，已按你的选择拦下，没有调用任何生成模型`,
      tone: 'border-stone-200 bg-stone-50 text-stone-700',
    }
  }
  const history =
    egress.history_messages > 0 ? `，含 ${egress.history_messages} 条历史消息` : ''
  return {
    title: '已发送到云端',
    detail: `${egress.chunks} 段 / ${egress.files} 个文件 / ${egress.chars} 字符（已含记忆）${history}`,
    tone: 'border-sky-200 bg-sky-50 text-sky-800',
  }
}

type Props = {
  message: ChatMessage | null
  onClose: () => void
}

export default function CitationPanel({ message, onClose }: Props) {
  return (
    <aside className="flex w-80 shrink-0 flex-col border-l border-stone-200 bg-white/60">
      <div className="flex items-center justify-between border-b border-stone-200 px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold text-stone-800">引用片段</h2>
          <p className="mt-0.5 text-xs text-stone-500">回答依据的原文，可对照行号核对</p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-md px-2 py-1 text-xs text-stone-500 transition hover:bg-stone-100 hover:text-stone-800"
        >
          收起
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-3">
        {message?.egress && (
          <div className={`mb-3 rounded-xl border px-3 py-2 ${receipt(message.egress).tone}`}>
            <p className="text-[11px] font-semibold">{receipt(message.egress).title}</p>
            <p className="mt-0.5 text-[10px] leading-relaxed">{receipt(message.egress).detail}</p>
            {message.egress.blocked_chunks > 0 && (
              <p className="mt-1 text-[10px] font-medium">
                另有 {message.egress.blocked_chunks} 段来自「禁止出站」的资料源，已剔除
              </p>
            )}
          </div>
        )}
        {!message && <p className="px-1 text-xs text-stone-500">点击回答中的编号可查看对应片段。</p>}
        {message && message.citations.length === 0 && message.memories.length === 0 && (
          <p className="px-1 text-xs text-stone-500">这次回答没有召回任何片段。</p>
        )}
        {message && message.memories.length > 0 && (
          <section className="mb-3">
            <h3 className="px-1 text-[11px] font-semibold text-stone-600">
              召回的记忆（{message.memories.length}）
            </h3>
            <p className="px-1 pb-1.5 text-[10px] text-stone-400">
              来自之前的对话，不是资料片段，因此没有编号
            </p>
            <ul className="space-y-2">
              {message.memories.map((memory) => (
                <li
                  key={memory.id}
                  className="rounded-xl border border-amber-200 bg-amber-50/50 p-3"
                >
                  <div className="flex items-center gap-1.5">
                    <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-800">
                      {KIND_LABEL[memory.kind]}
                    </span>
                    <span className="truncate text-[10px] text-stone-500">{memory.key}</span>
                    <span className="ml-auto text-[10px] tabular-nums text-stone-400">
                      {memory.score.toFixed(2)}
                    </span>
                  </div>
                  <p className="mt-1.5 whitespace-pre-wrap break-words text-xs leading-relaxed text-stone-700">
                    {memory.content}
                  </p>
                  <p className="mt-1 text-[10px] tabular-nums text-stone-400">
                    {memoryDate(memory)}
                  </p>
                </li>
              ))}
            </ul>
          </section>
        )}
        {message && message.citations.length > 0 && (
          <h3 className="px-1 pb-1.5 text-[11px] font-semibold text-stone-600">
            资料片段（{message.citations.length}）
          </h3>
        )}
        <ul className="space-y-2">
          {message?.citations.map((citation) => {
            const cited = message.cited.includes(citation.index)
            return (
              <li
                key={citation.chunk_id}
                id={`citation-${message.id}-${citation.index}`}
                className={`rounded-xl border p-3 transition ${
                  cited ? 'border-stone-300 bg-white' : 'border-stone-200 bg-stone-50/60'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="inline-flex h-5 min-w-5 items-center justify-center rounded bg-stone-900 px-1 text-[10px] font-semibold text-white">
                    {citation.index}
                  </span>
                  <span
                    className={`text-[10px] ${cited ? 'text-emerald-600' : 'text-stone-400'}`}
                  >
                    {cited ? '已引用' : '仅召回'}
                  </span>
                </div>
                <p className="mt-1 text-[10px] tabular-nums text-stone-400">
                  {traceLabel(citation)}
                </p>
                <p className="mt-2 break-all text-[11px] font-medium text-stone-700">
                  {citation.path}
                  {citation.heading && <span className="text-stone-400"> › {citation.heading}</span>}
                </p>
                <p className="text-[10px] text-stone-400">
                  第 {citation.start_line}–{citation.end_line} 行
                </p>
                <p className="mt-2 max-h-48 overflow-y-auto whitespace-pre-wrap break-words text-xs leading-relaxed text-stone-600">
                  {citation.text}
                </p>
              </li>
            )
          })}
        </ul>
      </div>
    </aside>
  )
}