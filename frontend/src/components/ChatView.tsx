import { useEffect, useRef } from 'react'
import type { ReactNode } from 'react'
import type { ChatMessage, Citation, Track } from '../types'

type Props = {
  messages: ChatMessage[]
  input: string
  onInputChange: (value: string) => void
  onSend: () => void
  streaming: boolean
  activeId: string | null
  onSelectMessage: (id: string) => void
  notice: string | null
  track: Track
  skipEgress: boolean
  onSkipEgressChange: (value: boolean) => void
}

const EXAMPLES = [
  '混合检索为什么要结合 BM25？',
  '分块策略有哪些取舍？',
  '我们公司明年的预算是多少？',
]

function seconds(ms: number): string {
  return `${(ms / 1000).toFixed(ms < 1000 ? 2 : 1)}s`
}

function latencyLabel(message: ChatMessage): string | null {
  if (message.totalMs === null) return null
  const parts: string[] = []
  if (message.ttftMs !== null) parts.push(`首 token ${seconds(message.ttftMs)}`)
  parts.push(`总 ${seconds(message.totalMs)}`)
  return parts.join(' · ')
}

function renderContent(
  content: string,
  citations: Citation[],
  onCite: (index: number) => void,
): ReactNode[] {
  return content.split(/(\[\d{1,2}\])/g).map((part, position) => {
    const match = /^\[(\d{1,2})\]$/.exec(part)
    if (!match) return <span key={position}>{part}</span>
    const index = Number(match[1])
    const exists = citations.some((citation) => citation.index === index)
    if (!exists) return <span key={position}>{part}</span>
    return (
      <button
        key={position}
        type="button"
        onClick={() => onCite(index)}
        title={`查看引用 ${index}`}
        className="mx-0.5 -translate-y-1 rounded bg-stone-200 px-1 py-px align-super text-[10px] font-semibold text-stone-700 transition hover:bg-stone-800 hover:text-white"
      >
        {index}
      </button>
    )
  })
}

export default function ChatView({
  messages,
  input,
  onInputChange,
  onSend,
  streaming,
  activeId,
  onSelectMessage,
  notice,
  track,
  skipEgress,
  onSkipEgressChange,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const node = scrollRef.current
    if (node) node.scrollTop = node.scrollHeight
  }, [messages])

  const hint =
    track === 'local'
      ? '本地模式：生成、检索与索引都在本机完成，零出站；重排在 CPU 上运行，不与本地模型抢显存。'
      : skipEgress
        ? '本次只做检索：命中的片段会显示在右侧，但不会发送给任何外部服务。'
        : '云端模式：只发送问题与命中的片段；原始文档不出本机，可在左侧按资料源禁止出站。'

  return (
    <section className="flex min-w-0 flex-1 flex-col">
      {notice && (
        <div className="border-b border-amber-200 bg-amber-50 px-6 py-2 text-xs text-amber-800">
          {notice}
        </div>
      )}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.length === 0 && (
            <div className="mt-16 rounded-2xl border border-stone-200 bg-white/70 p-6">
              <h2 className="text-base font-semibold text-stone-800">问点关于你资料的问题</h2>
              <p className="mt-1 text-sm text-stone-500">
                回答只基于已索引的文档，并给出文件名、小节和行号；资料里没有的，它会直接说找不到。
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                {EXAMPLES.map((example) => (
                  <button
                    key={example}
                    type="button"
                    onClick={() => onInputChange(example)}
                    className="rounded-full border border-stone-200 bg-white px-3 py-1.5 text-xs text-stone-600 transition hover:border-stone-400"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => {
            const isUser = message.role === 'user'
            const hasCitations = message.citations.length > 0
            const timing = isUser ? null : latencyLabel(message)
            return (
              <div key={message.id} className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                    isUser
                      ? 'bg-stone-900 text-stone-50'
                      : 'border border-stone-200 bg-white text-stone-800'
                  }`}
                >
                  {!isUser && (
                    <div className="mb-1.5 flex flex-wrap items-center gap-2 text-[11px] font-medium text-stone-400">
                      <span>LocalMind</span>
                      {message.track && (
                        <span
                          className={`rounded px-1.5 py-0.5 ${
                            message.track === 'local'
                              ? 'bg-emerald-50 text-emerald-700'
                              : 'bg-sky-50 text-sky-700'
                          }`}
                        >
                          {message.track === 'local' ? '本地 · 零出站' : '云端'}
                        </span>
                      )}
                      {message.refused && (
                        <span className="rounded bg-amber-100 px-1.5 py-0.5 text-amber-700">
                          {message.blocked ? '出站拦截' : '拒答'}
                        </span>
                      )}
                      {message.skipped && (
                        <span className="rounded bg-stone-100 px-1.5 py-0.5 text-stone-600">
                          仅检索
                        </span>
                      )}
                      {hasCitations && (
                        <button
                          type="button"
                          onClick={() => onSelectMessage(message.id)}
                          className={`rounded px-1.5 py-0.5 transition ${
                            activeId === message.id
                              ? 'bg-stone-800 text-white'
                              : 'hover:bg-stone-100 hover:text-stone-600'
                          }`}
                        >
                          引用 {message.citations.length}
                        </button>
                      )}
                      {timing && (
                        <span className="ml-auto text-[10px] tabular-nums text-stone-400">
                          {timing}
                        </span>
                      )}
                    </div>
                  )}

                  <p className="whitespace-pre-wrap break-words">
                    {renderContent(message.content, message.citations, (index) => {
                      onSelectMessage(message.id)
                      const target = document.getElementById(`citation-${message.id}-${index}`)
                      target?.scrollIntoView({ block: 'center', behavior: 'smooth' })
                    })}
                    {message.streaming && (
                      <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-stone-400 align-text-bottom" />
                    )}
                  </p>

                  {message.error && (
                    <p className="mt-2 rounded-lg bg-red-50 px-2.5 py-2 text-xs text-red-700">
                      {message.error}
                    </p>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <div className="border-t border-stone-200 bg-white/70 px-6 py-4">
        <div className="mx-auto max-w-3xl">
          <div className="rounded-2xl border border-stone-300 bg-white p-2 shadow-sm focus-within:border-stone-500">
            <div className="flex items-end gap-2">
              <textarea
                value={input}
                onChange={(event) => onInputChange(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault()
                    onSend()
                  }
                }}
                rows={2}
                placeholder="用日常语言提问，Enter 发送，Shift+Enter 换行"
                className="max-h-40 min-h-[44px] flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-stone-400"
              />
              <button
                type="button"
                onClick={onSend}
                disabled={streaming || !input.trim()}
                className="rounded-xl bg-stone-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-stone-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {streaming ? '生成中…' : '发送'}
              </button>
            </div>
            <div className="flex items-center gap-2 border-t border-stone-100 px-2 pt-1.5">
              <label
                className={`flex items-center gap-1.5 text-[11px] ${
                  track === 'local' ? 'text-stone-300' : 'cursor-pointer text-stone-600'
                }`}
                title="勾选后本次只检索，不调用任何生成模型；每轮都会重置"
              >
                <input
                  type="checkbox"
                  checked={track === 'local' ? true : skipEgress}
                  disabled={track === 'local'}
                  onChange={(event) => onSkipEgressChange(event.target.checked)}
                  className="h-3 w-3 accent-stone-800"
                />
                本次不上传（仅检索）
              </label>
            </div>
          </div>
          <p className="mt-1.5 text-center text-[11px] text-stone-400">{hint}</p>
        </div>
      </div>
    </section>
  )
}