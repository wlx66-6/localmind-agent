import type { ChatMessage } from '../types'

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
        {!message && <p className="px-1 text-xs text-stone-500">点击回答中的编号可查看对应片段。</p>}
        {message && message.citations.length === 0 && (
          <p className="px-1 text-xs text-stone-500">这次回答没有召回任何片段。</p>
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
                    {cited ? '已引用' : '仅召回'} · 相似度 {citation.score.toFixed(3)}
                  </span>
                </div>
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