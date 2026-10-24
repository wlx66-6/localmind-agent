import { useCallback, useEffect, useMemo, useState } from 'react'
import * as api from './api'
import ChatView from './components/ChatView'
import CitationPanel from './components/CitationPanel'
import Header from './components/Header'
import SourcesPanel from './components/SourcesPanel'
import type { ChatMessage, Health, Source } from './types'

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)

  const [sources, setSources] = useState<Source[]>([])
  const [sourcesLoading, setSourcesLoading] = useState(true)
  const [sourceError, setSourceError] = useState<string | null>(null)
  const [sourceNotice, setSourceNotice] = useState<string | null>(null)
  const [busyId, setBusyId] = useState<number | null>(null)

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [conversationId, setConversationId] = useState<number | null>(null)
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [activeId, setActiveId] = useState<string | null>(null)

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

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const result = await api.getHealth()
        if (cancelled) return
        setHealth(result)
        setHealthError(null)
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
        cited: [],
        refused: false,
        streaming: false,
      },
      {
        id: assistantId,
        role: 'assistant',
        content: '',
        citations: [],
        cited: [],
        refused: false,
        streaming: true,
      },
    ])
    setInput('')
    setStreaming(true)
    setActiveId(assistantId)

    let nextConversationId = conversationId
    const patch = (value: Partial<ChatMessage>) =>
      setMessages((prev) =>
        prev.map((message) =>
          message.id === assistantId ? { ...message, ...value } : message,
        ),
      )

    try {
      await api.streamChat(
        { message: question, conversation_id: conversationId },
        {
          onCitations: (id, items) => {
            nextConversationId = id
            patch({ citations: items })
          },
          onToken: (text) =>
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantId
                  ? { ...message, content: message.content + text }
                  : message,
              ),
            ),
          onDone: ({ conversationId: id, cited, refused }) => {
            nextConversationId = id
            patch({ cited, refused, streaming: false })
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
  }

  const cloudNotice =
    health && !health.deepseek_configured
      ? '云端密钥未配置：检索和引用可用，生成回答前请在 .env 中填写 DEEPSEEK_API_KEY 并重启后端。'
      : null

  return (
    <div className="flex h-full flex-col">
      <Header
        health={health}
        healthError={healthError}
        onNewChat={handleNewChat}
        canNewChat={messages.length > 0}
      />
      <div className="flex min-h-0 flex-1">
        <div className="hidden md:flex">
          <SourcesPanel
            sources={sources}
            loading={sourcesLoading}
            error={sourceError}
            notice={sourceNotice}
            busyId={busyId}
            onAdd={handleAdd}
            onScan={handleScan}
            onDelete={handleDelete}
          />
        </div>
        <ChatView
          messages={messages}
          input={input}
          onInputChange={setInput}
          onSend={() => void handleSend()}
          streaming={streaming}
          activeId={activeId}
          onSelectMessage={setActiveId}
          notice={cloudNotice}
        />
        {activeMessage && activeMessage.citations.length > 0 && (
          <div className="hidden lg:flex">
            <CitationPanel message={activeMessage} onClose={() => setActiveId(null)} />
          </div>
        )}
      </div>
    </div>
  )
}