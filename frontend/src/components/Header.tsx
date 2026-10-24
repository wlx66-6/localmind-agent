import type { Health, Track } from '../types'

type Props = {
  health: Health | null
  healthError: string | null
  track: Track
  onTrackChange: (track: Track) => void
  trackLocked: boolean
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

const TRACKS: { value: Track; label: string; title: string }[] = [
  { value: 'cloud', label: '云端', title: '问题与命中的片段发往 DeepSeek，质量高' },
  { value: 'local', label: '本地 · 零出站', title: 'Ollama 在本机生成，断网可用，数据不出本机' },
]

export default function Header({
  health,
  healthError,
  track,
  onTrackChange,
  trackLocked,
  onNewChat,
  canNewChat,
}: Props) {
  const localReady = health?.local_model ?? false
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
            <Pill
              label={localReady ? '本地模型 · 就绪' : '本地模型 · 未就绪'}
              ok={localReady}
              warnOnly
            />
          </>
        )}
        <span className="mx-1 h-5 w-px bg-stone-200" />
        <div
          className="flex rounded-lg border border-stone-300 bg-white p-0.5 text-xs"
          role="group"
          aria-label="生成轨道"
        >
          {TRACKS.map((option) => (
            <button
              key={option.value}
              type="button"
              title={option.title}
              aria-pressed={track === option.value}
              onClick={() => onTrackChange(option.value)}
              disabled={trackLocked}
              className={`rounded-md px-2.5 py-1 font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${
                track === option.value
                  ? 'bg-stone-900 text-white'
                  : 'text-stone-600 hover:bg-stone-100'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
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