import type { Health } from '../types'

type Props = {
  health: Health | null
  healthError: string | null
  onNewChat: () => void
  canNewChat: boolean
}

function Pill({ label, ok, warnOnly = false }: { label: string; ok: boolean; warnOnly?: boolean }) {
  const tone = ok
    ? 'bg-emerald-50 text-emerald-700 ring-emerald-200'
    : warnOnly
      ? 'bg-amber-50 text-amber-700 ring-amber-200'
      : 'bg-red-50 text-red-700 ring-red-200'
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${tone}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? 'bg-emerald-500' : 'bg-current'}`} />
      {label}
    </span>
  )
}

export default function Header({ health, healthError, onNewChat, canNewChat }: Props) {
  return (
    <header className="flex items-center justify-between border-b border-stone-200 bg-white/80 px-5 py-3 backdrop-blur">
      <div className="flex items-baseline gap-3">
        <h1 className="text-lg font-semibold tracking-tight">LocalMind</h1>
        <span className="text-xs text-stone-500">本地优先的个人知识代理</span>
      </div>

      <div className="flex items-center gap-2">
        {healthError && <Pill label="后端未连接" ok={false} />}
        {health && (
          <>
            <Pill label="Ollama" ok={health.ollama} />
            <Pill label="Qdrant" ok={health.qdrant} />
            <Pill
              label={health.deepseek_configured ? '云端 · 已配置' : '云端 · 未配置'}
              ok={health.deepseek_configured}
              warnOnly
            />
          </>
        )}
        <span className="mx-1 h-5 w-px bg-stone-200" />
        <button
          type="button"
          onClick={onNewChat}
          disabled={!canNewChat}
          className="rounded-lg border border-stone-300 bg-white px-3 py-1.5 text-xs font-medium text-stone-700 transition hover:border-stone-400 hover:bg-stone-50 disabled:cursor-not-allowed disabled:opacity-40"
        >
          新对话
        </button>
      </div>
    </header>
  )
}