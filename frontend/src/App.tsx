import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as api from './api'
import ChatView from './components/ChatView'
import CitationPanel from './components/CitationPanel'
import Header from './components/Header'
import MemoryPanel from './components/MemoryPanel'
import SourcesPanel from './components/SourcesPanel'
import type { ChatMessage, Health, MemoryItem, Source, Track } from './types'

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function blankAssistant(id: string, track: Track): ChatMessage {
  return {
    id,
    role: 'assistant',
    content: '',
    citations: [],
    memories: [],
    cited: [],
    refused: false,
    track,
    egress: null,
    retrieval: null,
    ttftMs: null,
    totalMs: null,
    skipped: false,
    blocked: false,
    streaming: true,
  }
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)

  const [sources, setSources] = useState<Source[]>([])
  const [sourcesLoading, setSourcesLoading] = useState(true)
  const [sourceError, setSourceError] = useState<string | null>(null)
  const [sourceNotice, setSourceNotice] = useState<string | null>(null)
  const [busyId, setBusyId] = useState<number | null>(null)

  const [memories, setMemories] = useState<MemoryItem[]>([])
  const [memoriesLoading, setMemoriesLoading] = useState(true)
  const [memoryError, setMemoryError] = useState<string | null>(null)
  const [memoryNotice, setMemoryNotice] = useState<string | null>(null)
  const [busyMemoryId, setBusyMemoryId] = useState<number | null>(null)
  const [includeSuperseded, setIncludeSuperseded] = useState(false)

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [conversationId, setConversationId] = useState<number | null>(null)
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [track, setTrack] = useState<Track>('cloud')
  // The composer switch is deliberately per-send: it resets after each turn so
  // "upload nothing" can never become a silently sticky setting.
  const [skipEgress, setSkipEgress] = useState(false)
  const trackTouched = useRef(false)

  const refreshSources = useCallback(async () => {
    try {
      setSources(await api.listSources())
      setSourceError(null)
    } catch (error) {
      setSourceError(errorText(error))
    } finally {
      setSourcesLoading(false)
    }
  }, [])

  useEffect(() => {
    void refreshSources()
  }, [refreshSources])

  const refreshMemories = useCallback(async () => {
    try {
      setMemories(await api.listMemories(includeSuperseded))
      setMemoryError(null)
    } catch (error) {
      setMemoryError(errorText(error))
    } finally {
      setMemoriesLoading(false)
    }
  }, [includeSuperseded])

  useEffect(() => {
    void refreshMemories()
  }, [refreshMemories])

  // Extraction runs in the background after a turn ends, so the panel cannot
  // refresh on `done` alone; keep a ref so the timer always uses the current
  // filter without re-subscribing the stream.
  const refreshMemoriesRef = useRef(refreshMemories)
  useEffect(() => {
    refreshMemoriesRef.current = refreshMemories
  }, [refreshMemories])

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const result = await api.getHealth()
        if (cancelled) return
        setHealth(result)
        setHealthError(null)
        if (!trackTouched.current) setTrack(result.default_track)
      } catch (error) {
        if (cancelled) return
        setHealth(null)
        setHealthError(errorText(error))
      }
    }
    void load()
    const timer = window.setInterval(() => void load(), 15000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  const activeMessage = useMemo(
    () => messages.find((message) => message.id === activeId) ?? null,
    [messages, activeId],
  )

  function handleTrackChange(next: Track) {
    trackTouched.current = true
    setTrack(next)
  }

  async function handleAdd(path: string): Promise<boolean> {
    setSourceError(null)
    setSourceNotice(null)
    try {
      const source = await api.addSource(path)
      setSourceNotice(`已添加，点击「扫描」建立索引：${source.path}`)
      await refreshSources()
      return true
    } catch (error) {
      setSourceError(errorText(error))
      return false
    }
  }

  async function handleScan(id: number) {
    setBusyId(id)
    setSourceError(null)
    setSourceNotice(null)
    try {
      const stats = await api.scanSource(id)
      setSourceNotice(
        `扫描完成：识别 ${stats.scanned}，索引 ${stats.indexed}，跳过 ${stats.skipped}，移除 ${stats.removed}，失败 ${stats.failed}`,
      )
      if (stats.errors.length > 0) setSourceError(stats.errors.slice(0, 3).join('；'))
      await refreshSources()
    } catch (error) {
      setSourceError(errorText(error))
    } finally {
      setBusyId(null)
    }
  }

  async function handleDelete(id: number) {
    setBusyId(id)
    setSourceError(null)
    setSourceNotice(null)
    try {
      await api.deleteSource(id)
      setSourceNotice('资料源已删除，对应向量也已清理')
      await refreshSources()
    } catch (error) {
      setSourceError(errorText(error))
    } finally {
      setBusyId(null)
    }
  }

  async function handleDeleteMemory(id: number) {
    setBusyMemoryId(id)
    setMemoryError(null)
    setMemoryNotice(null)
    try {
      await api.deleteMemory(id)
      setMemoryNotice('记忆已删除，后续回答不会再引用它')
      await refreshMemories()
    } catch (error) {
      setMemoryError(errorText(error))
    } finally {
      setBusyMemoryId(null)
    }
  }

  async function handleToggleEgress(id: number, allowed: boolean) {
    setBusyId(id)
    setSourceError(null)
    setSourceNotice(null)
    try {
      await api.updateSourceEgress(id, allowed)
      setSourceNotice(
        allowed ? '已允许该目录的片段出站' : '已禁止该目录出站：此后云端模式不会再发送它的片段',
      )
      await refreshSources()
    } catch (error) {
      setSourceError(errorText(error))
    } finally {
      setBusyId(null)
    }
  }

  async function handleSend() {
    const question = input.trim()
    if (!question || streaming) return

    const assistantId = crypto.randomUUID()
    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: 'user',
        content: question,
        citations: [],
        memories: [],
        cited: [],
        refused: false,
        track,
        egress: null,
        retrieval: null,
        ttftMs: null,
        totalMs: null,
        skipped: false,
        blocked: false,
        streaming: false,
      },
      blankAssistant(assistantId, track),
    ])
    setInput('')
    setStreaming(true)
    setActiveId(assistantId)
    setSkipEgress(false)

    let nextConversationId = conversationId
    const patch = (value: Partial<ChatMessage>) =>
      setMessages((prev) =>
        prev.map((message) =>
          message.id === assistantId ? { ...message, ...value } : message,
        ),
      )

    try {
      await api.streamChat(
        {
          message: question,
          conversation_id: conversationId,
          track,
          skip_egress: skipEgress,
        },
        {
          onCitations: (payload) => {
            nextConversationId = payload.conversationId
            patch({
              track: payload.track,
              citations: payload.items,
              memories: payload.memories,
              egress: payload.egress,
              retrieval: payload.retrieval,
            })
          },
          onToken: (text) =>
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantId
                  ? { ...message, content: message.content + text }
                  : message,
              ),
            ),
          onDone: (result) => {
            nextConversationId = result.conversationId
            patch({
              cited: result.cited,
              refused: result.refused,
              streaming: false,
              track: result.track,
              skipped: result.skipped,
              blocked: result.blocked,
              ttftMs: result.ttftMs,
              totalMs: result.totalMs,
            })
            // Background extraction usually lands a few seconds later; the
            // panel also has a manual refresh.
            window.setTimeout(() => void refreshMemoriesRef.current(), 4000)
          },
          onError: (message) => patch({ error: message, streaming: false }),
        },
      )
    } catch (error) {
      patch({ error: errorText(error), streaming: false })
    } finally {
      setStreaming(false)
      setConversationId(nextConversationId)
      patch({ streaming: false })
    }
  }

  function handleNewChat() {
    if (streaming) return
    setMessages([])
    setConversationId(null)
    setActiveId(null)
    setInput('')
    setSkipEgress(false)
  }

  const notice =
    track === 'cloud' && health && !health.deepseek_configured
      ? '云端密钥未配置：检索和引用可用，生成回答前请在 .env 中填写 DEEPSEEK_API_KEY 并重启后端。'
      : track === 'local' && health && !health.local_model
        ? '本地模型未就绪：请先 ollama pull qwen3.5:4b，再重试本地模式。'
        : null

  return (
    <div className="flex h-full flex-col">
      <Header
        health={health}
        healthError={healthError}
        track={track}
        onTrackChange={handleTrackChange}
        trackLocked={streaming}
        onNewChat={handleNewChat}
        canNewChat={messages.length > 0}
      />
      <div className="flex min-h-0 flex-1">
        <aside className="hidden w-72 shrink-0 flex-col border-r border-stone-200 bg-white/60 md:flex">
          <SourcesPanel
            sources={sources}
            loading={sourcesLoading}
            error={sourceError}
            notice={sourceNotice}
            busyId={busyId}
            onAdd={handleAdd}
            onScan={handleScan}
            onDelete={handleDelete}
            onToggleEgress={(id, allowed) => void handleToggleEgress(id, allowed)}
          />
          <MemoryPanel
            memories={memories}
            loading={memoriesLoading}
            error={memoryError}
            notice={memoryNotice}
            includeSuperseded={includeSuperseded}
            onIncludeSupersededChange={setIncludeSuperseded}
            onRefresh={() => void refreshMemories()}
            onDelete={(id) => void handleDeleteMemory(id)}
            busyId={busyMemoryId}
          />
        </aside>
        <ChatView
          messages={messages}
          input={input}
          onInputChange={setInput}
          onSend={() => void handleSend()}
          streaming={streaming}
          activeId={activeId}
          onSelectMessage={setActiveId}
          notice={notice}
          track={track}
          skipEgress={skipEgress}
          onSkipEgressChange={setSkipEgress}
        />
        {activeMessage &&
          (activeMessage.citations.length > 0 || activeMessage.memories.length > 0) && (
          <div className="hidden lg:flex">
            <CitationPanel message={activeMessage} onClose={() => setActiveId(null)} />
          </div>
        )}
      </div>
    </div>
  )
}