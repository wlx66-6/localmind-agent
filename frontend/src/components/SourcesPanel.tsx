import { useState } from 'react'
import type { Source } from '../types'

type Props = {
  sources: Source[]
  loading: boolean
  error: string | null
  notice: string | null
  busyId: number | null
  onAdd: (path: string) => Promise<boolean>
  onScan: (id: number) => Promise<void>
  onDelete: (id: number) => Promise<void>
  onToggleEgress: (id: number, allowed: boolean) => void
}

function formatTime(value: string | null): string {
  if (!value) return '尚未扫描'
  return new Date(value).toLocaleString('zh-CN', { hour12: false })
}

export default function SourcesPanel({
  sources,
  loading,
  error,
  notice,
  busyId,
  onAdd,
  onScan,
  onDelete,
  onToggleEgress,
}: Props) {
  const [path, setPath] = useState('')
  const [adding, setAdding] = useState(false)

  async function handleAdd() {
    const value = path.trim()
    if (!value || adding) return
    setAdding(true)
    try {
      if (await onAdd(value)) setPath('')
    } finally {
      setAdding(false)
    }
  }

  return (
    <section className="flex min-h-0 flex-1 flex-col">
      <div className="border-b border-stone-200 px-4 py-3">
        <h2 className="text-sm font-semibold text-stone-800">资料源</h2>
        <p className="mt-0.5 text-xs text-stone-500">
          只读取白名单目录内的 md / txt 文件；关掉「出站」后，该目录的片段不会进入云端请求
        </p>
      </div>

      <div className="space-y-2 border-b border-stone-200 px-4 py-3">
        <input
          value={path}
          onChange={(event) => setPath(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') void handleAdd()
          }}
          placeholder="绝对路径，例如 D:\notes"
          className="w-full rounded-lg border border-stone-300 bg-white px-2.5 py-1.5 text-xs outline-none transition focus:border-stone-500"
        />
        <button
          type="button"
          onClick={() => void handleAdd()}
          disabled={adding || !path.trim()}
          className="w-full rounded-lg bg-stone-900 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-stone-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {adding ? '添加中…' : '添加目录'}
        </button>
        {error && <p className="text-xs text-red-600">{error}</p>}
        {notice && <p className="text-xs text-stone-500">{notice}</p>}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
        {loading && <p className="px-1 text-xs text-stone-500">加载中…</p>}
        {!loading && sources.length === 0 && (
          <p className="px-1 text-xs text-stone-500">还没有资料源。先添加一个目录，再扫描建立索引。</p>
        )}
        <ul className="space-y-2">
          {sources.map((source) => (
            <li key={source.id} className="rounded-xl border border-stone-200 bg-white p-3">
              <p className="break-all text-xs font-medium text-stone-800">{source.path}</p>
              <p className="mt-1 text-[11px] text-stone-500">
                {source.document_count} 个文档 · {source.chunk_count} 个分块
              </p>
              <p className="mt-0.5 text-[11px] text-stone-400">{formatTime(source.last_scan_at)}</p>
              <label
                className={`mt-2 flex items-center gap-1.5 text-[11px] ${
                  source.egress_allowed ? 'text-stone-500' : 'font-medium text-amber-700'
                }`}
                title="关闭后，云端模式检索到该目录的片段会被服务端剔除，本地模式不受影响"
              >
                <input
                  type="checkbox"
                  checked={source.egress_allowed}
                  disabled={busyId === source.id}
                  onChange={(event) => onToggleEgress(source.id, event.target.checked)}
                  className="h-3 w-3 accent-stone-800"
                />
                {source.egress_allowed ? '允许片段出站' : '禁止出站（仅本地可用）'}
              </label>
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  onClick={() => void onScan(source.id)}
                  disabled={busyId === source.id}
                  className="rounded-md border border-stone-300 px-2 py-1 text-[11px] font-medium text-stone-700 transition hover:bg-stone-50 disabled:opacity-40"
                >
                  {busyId === source.id ? '扫描中…' : '扫描'}
                </button>
                <button
                  type="button"
                  onClick={() => void onDelete(source.id)}
                  disabled={busyId === source.id}
                  className="rounded-md px-2 py-1 text-[11px] font-medium text-red-600 transition hover:bg-red-50 disabled:opacity-40"
                >
                  删除
                </button>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}