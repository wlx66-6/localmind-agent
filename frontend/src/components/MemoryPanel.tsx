import { localDate } from '../format'
import type { MemoryItem } from '../types'

type Props = {
  memories: MemoryItem[]
  loading: boolean
  error: string | null
  notice: string | null
  includeSuperseded: boolean
  onIncludeSupersededChange: (value: boolean) => void
  onRefresh: () => void
  onDelete: (id: number) => void
  busyId: number | null
}

const KIND_LABEL: Record<MemoryItem['kind'], string> = {
  fact: '事实',
  preference: '偏好',
  summary: '摘要',
}


/** Compact list of what the agent remembers, with deletion as the only edit. */
export default function MemoryPanel({
  memories,
  loading,
  error,
  notice,
  includeSuperseded,
  onIncludeSupersededChange,
  onRefresh,
  onDelete,
  busyId,
}: Props) {
  return (
    <section className="flex max-h-[45%] min-h-0 shrink-0 flex-col border-t border-stone-200">
      <div className="border-b border-stone-200 px-4 py-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-stone-800">长期记忆</h2>
          <button
            type="button"
            onClick={onRefresh}
            className="rounded-md px-2 py-0.5 text-[11px] text-stone-500 transition hover:bg-stone-100 hover:text-stone-800"
          >
            刷新
          </button>
        </div>
        <p className="mt-0.5 text-xs text-stone-500">
          从对话里抽取的事实与偏好，跨会话召回；删除后不再参与回答
        </p>
        <label className="mt-2 flex cursor-pointer items-center gap-1.5 text-[11px] text-stone-500">
          <input
            type="checkbox"
            checked={includeSuperseded}
            onChange={(event) => onIncludeSupersededChange(event.target.checked)}
            className="h-3 w-3 accent-stone-800"
          />
          显示已被替换的旧值
        </label>
        {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
        {notice && <p className="mt-2 text-xs text-stone-500">{notice}</p>}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
        {loading && <p className="px-1 text-xs text-stone-500">加载中…</p>}
        {!loading && memories.length === 0 && (
          <p className="px-1 text-xs text-stone-500">
            还没有记忆。聊几轮之后，它会把你提到的事实和偏好记下来。
          </p>
        )}
        <ul className="space-y-2">
          {memories.map((memory) => {
            const superseded = memory.status === 'superseded'
            return (
              <li
                key={memory.id}
                className={`rounded-xl border p-3 ${
                  superseded
                    ? 'border-dashed border-stone-200 bg-stone-50/60'
                    : 'border-stone-200 bg-white'
                }`}
              >
                <div className="flex items-center gap-1.5">
                  <span className="rounded bg-stone-100 px-1.5 py-0.5 text-[10px] font-medium text-stone-600">
                    {KIND_LABEL[memory.kind]}
                  </span>
                  {memory.kind !== 'summary' && (
                    <span className="truncate text-[11px] font-medium text-stone-700">
                      {memory.key}
                    </span>
                  )}
                  {superseded && (
                    <span className="ml-auto rounded bg-amber-50 px-1.5 py-0.5 text-[10px] text-amber-700">
                      已替换
                    </span>
                  )}
                  {!superseded && !memory.embedding_ok && (
                    <span
                      className="ml-auto rounded bg-amber-50 px-1.5 py-0.5 text-[10px] text-amber-700"
                      title="向量写入失败，这条记忆无法被语义召回"
                    >
                      未入索引
                    </span>
                  )}
                </div>
                <p
                  className={`mt-1.5 whitespace-pre-wrap break-words text-xs leading-relaxed ${
                    superseded ? 'text-stone-400 line-through decoration-stone-300' : 'text-stone-700'
                  }`}
                >
                  {memory.content}
                </p>
                <div className="mt-1.5 flex items-center justify-between gap-2">
                  <p className="text-[10px] tabular-nums text-stone-400">
                    记于 {localDate(memory.created_at)}
                    {memory.occurred_at && ` · 事发 ${memory.occurred_at}`}
                  </p>
                  <button
                    type="button"
                    disabled={busyId === memory.id}
                    onClick={() => {
                      if (window.confirm('删除这条记忆？删除后它不会再出现在回答依据里。')) {
                        onDelete(memory.id)
                      }
                    }}
                    className="rounded-md px-1.5 py-0.5 text-[10px] font-medium text-red-600 transition hover:bg-red-50 disabled:opacity-40"
                  >
                    删除
                  </button>
                </div>
              </li>
            )
          })}
        </ul>
      </div>
    </section>
  )
}