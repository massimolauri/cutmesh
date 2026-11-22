"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  CatalogModel,
  ChatMessageItem,
  CutmeshTelemetry
} from "@/types/control-plane";
import {
  API_BASE,
  createOpenAiApiKey,
  getStoredJwtToken
} from "@/services/control-plane-api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import { ChatMarkdown } from "@/components/ui/chat-markdown";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger
} from "@/components/ui/collapsible";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from "@/components/ui/tooltip";
import {
  AlertTriangle,
  Bot,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  CornerDownRight,
  Loader2,
  Plus,
  Send,
  Square,
  Trash2,
  User
} from "lucide-react";
import { CutmeshLogo } from "@/components/ui/cutmesh-logo";
import { useI18n } from "@/i18n/translations";

interface SavedChatSession {
  id: string;
  title: string;
  updatedAt: number;
  modelId: string;
  systemPrompt: string;
  messages: ChatMessageItem[];
  lastTelemetry: CutmeshTelemetry | null;
}

const STORAGE_KEY_SESSIONS = "cutmesh_playground_sessions_v1";
const STORAGE_KEY_ACTIVE_ID = "cutmesh_playground_active_session_v1";
const STORAGE_KEY_SETTINGS = "cutmesh_playground_settings_v2";

const DEFAULT_SYSTEM_PROMPT = "";

function createInitialSession(modelId: string, index = 1): SavedChatSession {
  return {
    id: `session-${String(index).padStart(2, "0")}-${Date.now().toString(36).slice(-4)}`,
    title: `Sessione #${index}`,
    updatedAt: Date.now(),
    modelId,
    systemPrompt: DEFAULT_SYSTEM_PROMPT,
    messages: [],
    lastTelemetry: null
  };
}

interface AiPlaygroundSectionProps {
  t: Record<string, string>;
  models: CatalogModel[];
  initialModelId?: string;
  latestRawApiKey: string;
  onSetLatestRawApiKey: (rawKey: string) => void;
  onRefresh: () => void;
}

export function AiPlaygroundSection({
  t,
  models,
  initialModelId,
  latestRawApiKey,
  onSetLatestRawApiKey,
  onRefresh
}: AiPlaygroundSectionProps) {
  const defaultModelId =
    initialModelId || models[0]?.modelId || "qwen3.8-27b";

  const [sessions, setSessions] = useState<SavedChatSession[]>(() => {
    if (typeof window === "undefined") {
      return [createInitialSession(defaultModelId, 1)];
    }
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY_SESSIONS);
      if (raw) {
        const parsed = JSON.parse(raw) as SavedChatSession[];
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed;
        }
      }
    } catch {
      // ignore corrupted storage
    }
    return [createInitialSession(defaultModelId, 1)];
  });

  const [activeSessionId, setActiveSessionId] = useState<string>(() => {
    if (typeof window !== "undefined") {
      const savedId = window.localStorage.getItem(STORAGE_KEY_ACTIVE_ID);
      if (savedId) return savedId;
    }
    return "";
  });

  const activeSession =
    sessions.find((s) => s.id === activeSessionId) || sessions[0];

  const [playgroundModel, setPlaygroundModel] = useState<string>(
    activeSession?.modelId || defaultModelId
  );
  const [systemPrompt, setSystemPrompt] = useState<string>(
    activeSession?.systemPrompt || DEFAULT_SYSTEM_PROMPT
  );
  const [temperature, setTemperature] = useState<number>(0.6);
  const [maxTokens, setMaxTokens] = useState<number>(8192);
  const [thinkingMode, setThinkingMode] = useState<"off" | "low" | "medium" | "xhigh">("low");
  const [chatInput, setChatInput] = useState<string>("");
  const [sendingChat, setSendingChat] = useState<boolean>(false);
  const { locale } = useI18n();
  const isEn = locale === "en";
  const [liveTokPerSec, setLiveTokPerSec] = useState<number>(0);
  const [livePromptTokPerSec, setLivePromptTokPerSec] = useState<number>(0);
  const [expandedThinkingMap, setExpandedThinkingMap] = useState<Record<number, boolean>>({});
  const [copiedMsgIdx, setCopiedMsgIdx] = useState<number | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Restore saved parameters on mount (with automatic upgrade for old low maxTokens)
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY_SETTINGS);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (typeof parsed.temperature === "number") setTemperature(parsed.temperature);
        if (typeof parsed.maxTokens === "number" && parsed.maxTokens >= 1024) {
          setMaxTokens(parsed.maxTokens);
        } else {
          setMaxTokens(8192);
        }
        if (parsed.thinkingMode) setThinkingMode(parsed.thinkingMode);
      }
    } catch {
      // ignore
    }
  }, []);

  // Persist parameters
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(
        STORAGE_KEY_SETTINGS,
        JSON.stringify({ temperature, maxTokens, thinkingMode })
      );
    } catch {
      // ignore
    }
  }, [temperature, maxTokens, thinkingMode]);

  // Ensure activeSessionId points to a valid session
  useEffect(() => {
    if (sessions.length > 0 && !sessions.some((s) => s.id === activeSessionId)) {
      setActiveSessionId(sessions[0].id);
    }
  }, [sessions, activeSessionId]);

  // Persist sessions and activeSessionId in localStorage
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(STORAGE_KEY_SESSIONS, JSON.stringify(sessions));
      if (activeSession?.id) {
        window.localStorage.setItem(STORAGE_KEY_ACTIVE_ID, activeSession.id);
      }
    } catch {
      // ignore storage quota errors
    }
  }, [sessions, activeSession?.id]);

  // Sync model and systemPrompt when switching sessions
  useEffect(() => {
    if (!activeSession) return;
    setSystemPrompt(activeSession.systemPrompt || DEFAULT_SYSTEM_PROMPT);
    if (activeSession.modelId && models.some((m) => m.modelId === activeSession.modelId)) {
      setPlaygroundModel(activeSession.modelId);
    }
    const lastAssistant = [...(activeSession.messages || [])]
      .reverse()
      .find((m) => m.role === "assistant");
    setLiveTokPerSec(lastAssistant?.tokPerSec || 0);
    setLivePromptTokPerSec(lastAssistant?.promptTokPerSec || 0);
  }, [activeSession?.id]);

  useEffect(() => {
    if (initialModelId) {
      setPlaygroundModel(initialModelId);
    }
  }, [initialModelId]);

  useEffect(() => {
    if (models.length > 0 && !models.some((m) => m.modelId === playgroundModel)) {
      setPlaygroundModel(models[0].modelId);
    }
  }, [models, playgroundModel]);

  const chatHistory = activeSession?.messages || [];
  const lastTelemetry = activeSession?.lastTelemetry || null;
  const conversationId = activeSession?.id || "session-01";
  const enableThinking = thinkingMode !== "off";

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory.length, sendingChat]);

  const updateSessionById = (
    sessionId: string,
    updater: (prev: SavedChatSession) => SavedChatSession
  ) => {
    setSessions((prev) =>
      prev.map((s) => (s.id === sessionId ? updater(s) : s))
    );
  };

  const handleCreateNewSession = () => {
    handleStopGeneration();
    const nextIdx = sessions.length + 1;
    const fresh = createInitialSession(playgroundModel, nextIdx);
    setSessions((prev) => [fresh, ...prev]);
    setActiveSessionId(fresh.id);
    setLiveTokPerSec(0);
    setLivePromptTokPerSec(0);
  };

  const handleClearCurrentSession = () => {
    if (!activeSession) return;
    handleStopGeneration();
    updateSessionById(activeSession.id, (s) => ({
      ...s,
      messages: [],
      lastTelemetry: null,
      updatedAt: Date.now()
    }));
    setLiveTokPerSec(0);
    setLivePromptTokPerSec(0);
  };

  const handleDeleteSession = (sessionId: string) => {
    handleStopGeneration();
    setSessions((prev) => {
      const remaining = prev.filter((s) => s.id !== sessionId);
      if (remaining.length === 0) {
        const fresh = createInitialSession(playgroundModel, 1);
        setActiveSessionId(fresh.id);
        return [fresh];
      }
      if (activeSessionId === sessionId) {
        setActiveSessionId(remaining[0].id);
      }
      return remaining;
    });
  };

  const toggleThinkingExpand = (idx: number) => {
    setExpandedThinkingMap((prev) => ({
      ...prev,
      [idx]: !(prev[idx] ?? false)
    }));
  };

  const handleCopyMessage = async (idx: number, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMsgIdx(idx);
      setTimeout(() => setCopiedMsgIdx(null), 1800);
    } catch {
      // ignore
    }
  };

  const handleStopGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  };

  const handleSendPlaygroundMessage = async (overridePrompt?: string) => {
    const promptToSend = (overridePrompt !== undefined ? overridePrompt : chatInput).trim();
    if (!promptToSend || sendingChat || !activeSession) return;
    const targetSessionId = activeSession.id;
    if (overridePrompt === undefined) {
      setChatInput("");
    }
    setSendingChat(true);
    setLiveTokPerSec(0);
    setLivePromptTokPerSec(0);

    const userMsg: ChatMessageItem = { role: "user", content: promptToSend };
    const currentMessages = activeSession.messages || [];
    const assistantIdx = currentMessages.length + 1;
    const placeholderAssistant: ChatMessageItem = {
      role: "assistant",
      content: "",
      reasoningContent: "",
      isThinking: enableThinking,
      isStreaming: true,
      completionTokens: 0,
      thinkingTokens: 0,
      tokPerSec: 0
    };

    const nextHistory = [...currentMessages, userMsg];
    const derivedTitle =
      currentMessages.length === 0
        ? promptToSend.length > 36
          ? `${promptToSend.slice(0, 36)}...`
          : promptToSend
        : activeSession.title;

    updateSessionById(targetSessionId, (s) => ({
      ...s,
      title: derivedTitle,
      modelId: playgroundModel,
      systemPrompt,
      updatedAt: Date.now(),
      messages: [...nextHistory, placeholderAssistant]
    }));
    setExpandedThinkingMap((prev) => ({ ...prev, [assistantIdx]: false }));

    const controller = new AbortController();
    abortControllerRef.current = controller;
    const reqStartMs = performance.now();
    let firstTokenMs: number | null = null;

    try {
      let bearerToken = latestRawApiKey || getStoredJwtToken() || "";
      if (!bearerToken) {
        const created = await createOpenAiApiKey("AI-Studio-Playground-Session", 120);
        if (created) {
          bearerToken = created.rawApiKeySecret;
          onSetLatestRawApiKey(bearerToken);
        }
      }

      // Build multi-turn OpenAI conversation messages (preserving full session memory)
      const apiMessages = [
        ...(systemPrompt.trim() ? [{ role: "system", content: systemPrompt.trim() }] : []),
        ...nextHistory
          .filter(
            (m) =>
              (m.content && m.content.trim().length > 0) ||
              (m.reasoningContent && m.reasoningContent.trim().length > 0)
          )
          .map((m) => ({
            role: m.role,
            content:
              m.content && m.content.trim().length > 0
                ? m.content.replace(/<think>[\s\S]*?<\/think>/g, "").trim()
                : m.reasoningContent?.trim() || ""
          }))
      ];

      const res = await fetch(`${API_BASE}/v1/chat/completions/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${bearerToken}`
        },
        body: JSON.stringify({
          model: playgroundModel,
          conversation_id: targetSessionId,
          temperature,
          max_tokens: maxTokens,
          stream: true,
          enable_thinking: enableThinking,
          reasoning_effort: enableThinking ? thinkingMode : "off",
          chat_template_kwargs: enableThinking
            ? { enable_thinking: true, reasoning_effort: thinkingMode }
            : { enable_thinking: false },
          messages: apiMessages
        }),
        signal: controller.signal
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        updateSessionById(targetSessionId, (s) => ({
          ...s,
          updatedAt: Date.now(),
          messages: [
            ...nextHistory,
            {
              role: "assistant",
              content: `[Errore ${res.status}] ${err.message || "Impossibile contattare il servizio AI attivo."}`,
              isStreaming: false
            }
          ]
        }));
        return;
      }

      const contentType = res.headers.get("content-type") || "";
      if (!contentType.includes("text/event-stream") || !res.body) {
        const data = await res.json();
        const choice = data.choices?.[0];
        const msg = choice?.message;
        const reply = msg?.content || "";
        const reasoning = msg?.reasoning_content || "";
        const finishReason = choice?.finish_reason || "stop";
        const promptTok = data.usage?.prompt_tokens || 0;
        const compTok = data.usage?.completion_tokens || 0;
        const totalTok = data.usage?.total_tokens || promptTok + compTok;
        const tel: CutmeshTelemetry | undefined = data.cutmesh_telemetry ?? data.synapse_telemetry;
        const tokSec = tel?.predicted_per_second || 0;
        const promptTokSec = tel?.prompt_per_second || 0;

        if (tel) {
          setLiveTokPerSec(tokSec);
          setLivePromptTokPerSec(promptTokSec);
        }
        updateSessionById(targetSessionId, (s) => ({
          ...s,
          updatedAt: Date.now(),
          lastTelemetry: tel || s.lastTelemetry,
          messages: [
            ...nextHistory,
            {
              role: "assistant",
              content: reply,
              reasoningContent: reasoning,
              isThinking: false,
              isStreaming: false,
              finishReason,
              latencyMs: tel?.latency_ms || Math.round(performance.now() - reqStartMs),
              promptMs: tel?.prompt_ms,
              predictedMs: tel?.predicted_ms,
              promptTokens: promptTok,
              completionTokens: compTok,
              thinkingTokens: tel?.thinking_tokens || 0,
              tokens: totalTok,
              tokPerSec: tokSec,
              promptTokPerSec: promptTokSec
            }
          ]
        }));
        onRefresh();
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let accContent = "";
      let accReasoning = "";
      let promptTokens = 0;
      let completionTokens = 0;
      let thinkingTokens = 0;
      let promptMs = 0;
      let predictedMs = 0;
      let serverTokPerSec = 0;
      let serverPromptTokPerSec = 0;
      let inInlineThinkTag = false;
      let finishReason: string | undefined = undefined;
      let latestTel: CutmeshTelemetry | null = null;

      const updateAssistantState = (streaming: boolean, stillThinking: boolean) => {
        const nowMs = performance.now();
        const elapsedTotalMs = Math.round(nowMs - reqStartMs);
        let effectiveTokPerSec = serverTokPerSec;
        if (effectiveTokPerSec <= 0 && firstTokenMs !== null && completionTokens > 1) {
          const decodeSec = Math.max(0.05, (nowMs - firstTokenMs) / 1000);
          effectiveTokPerSec = (completionTokens - 1) / decodeSec;
        }
        if (effectiveTokPerSec > 0) {
          setLiveTokPerSec(effectiveTokPerSec);
        }
        if (serverPromptTokPerSec > 0) {
          setLivePromptTokPerSec(serverPromptTokPerSec);
        }

        updateSessionById(targetSessionId, (s) => ({
          ...s,
          updatedAt: Date.now(),
          lastTelemetry: latestTel || s.lastTelemetry,
          messages: [
            ...nextHistory,
            {
              role: "assistant",
              content: accContent,
              reasoningContent: accReasoning,
              isThinking: stillThinking,
              isStreaming: streaming,
              finishReason,
              latencyMs: elapsedTotalMs,
              promptMs: promptMs > 0 ? promptMs : undefined,
              predictedMs: predictedMs > 0 ? predictedMs : undefined,
              promptTokens: promptTokens > 0 ? promptTokens : undefined,
              completionTokens,
              thinkingTokens,
              tokens: promptTokens + completionTokens,
              tokPerSec: effectiveTokPerSec > 0 ? effectiveTokPerSec : undefined,
              promptTokPerSec: serverPromptTokPerSec > 0 ? serverPromptTokPerSec : undefined
            }
          ]
        }));
      };

      const processSseLine = (rawLine: string) => {
        const line = rawLine.trim();
        if (!line.startsWith("data:")) return;
        const dataStr = line.slice(5).trim();
        if (!dataStr || dataStr === "[DONE]") return;

        try {
          const parsed = JSON.parse(dataStr);

          if (parsed.error?.message) {
            accContent += `\n[Errore Stream]: ${parsed.error.message}`;
            updateAssistantState(false, false);
            return;
          }

          if (parsed.cutmesh_telemetry || parsed.synapse_telemetry) {
            const tel: CutmeshTelemetry = parsed.cutmesh_telemetry ?? parsed.synapse_telemetry;
            latestTel = tel;
            if (tel.predicted_per_second && tel.predicted_per_second > 0) {
              serverTokPerSec = tel.predicted_per_second;
            }
            if (tel.prompt_per_second && tel.prompt_per_second > 0) {
              serverPromptTokPerSec = tel.prompt_per_second;
            }
            if (tel.prompt_ms && tel.prompt_ms > 0) {
              promptMs = tel.prompt_ms;
            }
            if (tel.predicted_ms && tel.predicted_ms > 0) {
              predictedMs = tel.predicted_ms;
            }
          }

          if (parsed.usage) {
            promptTokens = parsed.usage.prompt_tokens || promptTokens;
            completionTokens = parsed.usage.completion_tokens || completionTokens;
          }

          if (parsed.timings) {
            const tm = parsed.timings;
            if (typeof tm.prompt_n === "number" && tm.prompt_n > 0) {
              promptTokens = tm.prompt_n;
            }
            if (typeof tm.predicted_n === "number" && tm.predicted_n > 0) {
              completionTokens = tm.predicted_n;
            }
            if (typeof tm.prompt_ms === "number" && tm.prompt_ms > 0) {
              promptMs = tm.prompt_ms;
            }
            if (typeof tm.predicted_ms === "number" && tm.predicted_ms > 0) {
              predictedMs = tm.predicted_ms;
            }
            if (typeof tm.prompt_per_second === "number" && tm.prompt_per_second > 0) {
              serverPromptTokPerSec = tm.prompt_per_second;
            }
            if (typeof tm.predicted_per_second === "number" && tm.predicted_per_second > 0) {
              serverTokPerSec = tm.predicted_per_second;
            }
          }

          const choice0 = parsed.choices?.[0];
          if (choice0?.finish_reason) {
            finishReason = choice0.finish_reason;
          }

          const delta = choice0?.delta;
          if (delta) {
            const rDelta: string = delta.reasoning_content || "";
            const cDelta: string = delta.content || "";

            if (rDelta) {
              if (firstTokenMs === null) firstTokenMs = performance.now();
              accReasoning += rDelta;
              thinkingTokens += 1;
              if (completionTokens < thinkingTokens) {
                completionTokens = thinkingTokens;
              }
              updateAssistantState(true, true);
            }

            if (cDelta) {
              if (firstTokenMs === null) firstTokenMs = performance.now();
              if (cDelta.includes("<think>")) {
                inInlineThinkTag = true;
                const cleaned = cDelta.replace("<think>", "");
                accReasoning += cleaned;
                thinkingTokens += 1;
                updateAssistantState(true, true);
              } else if (cDelta.includes("</think>")) {
                inInlineThinkTag = false;
                const parts = cDelta.split("</think>");
                accReasoning += parts[0] || "";
                accContent += parts[1] || "";
                updateAssistantState(true, false);
              } else if (inInlineThinkTag) {
                accReasoning += cDelta;
                thinkingTokens += 1;
                updateAssistantState(true, true);
              } else {
                accContent += cDelta;
                if (!parsed.timings) {
                  completionTokens += 1;
                }
                updateAssistantState(true, false);
              }
            }
          }
        } catch {
          // ignore partial JSON
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          buffer += decoder.decode();
          if (buffer.trim()) {
            for (const remLine of buffer.split("\n")) {
              processSseLine(remLine);
            }
          }
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const rawLine of lines) {
          processSseLine(rawLine);
        }
      }

      updateAssistantState(false, false);
      onRefresh();
    } catch (err: any) {
      if (err?.name === "AbortError") {
        updateSessionById(targetSessionId, (s) => {
          const copy = [...s.messages];
          const last = copy[copy.length - 1];
          if (last && last.role === "assistant") {
            copy[copy.length - 1] = {
              ...last,
              isStreaming: false,
              isThinking: false
            };
          }
          return { ...s, messages: copy };
        });
      } else {
        updateSessionById(targetSessionId, (s) => ({
          ...s,
          messages: [
            ...nextHistory,
            {
              role: "assistant",
              content: `[Errore di connessione]: ${err?.message || "Richiesta interrotta"}`,
              isStreaming: false
            }
          ]
        }));
      }
    } finally {
      setSendingChat(false);
      abortControllerRef.current = null;
    }
  };

  return (
    <div className="max-w-4xl mx-auto flex flex-col h-[calc(100vh-8.5rem)] min-h-[560px]">
      {/* Minimal Top Session Bar */}
      <div className="flex items-center justify-between gap-2 pb-3 border-b border-border">
        <div className="flex items-center gap-2 min-w-0">
          <Select value={conversationId} onValueChange={setActiveSessionId}>
            <SelectTrigger className="h-8 w-[190px] text-xs border-border bg-card">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {sessions.map((s) => (
                <SelectItem key={s.id} value={s.id} className="text-xs">
                  {s.title} ({s.messages.length})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button
            variant="outline"
            size="sm"
            onClick={handleCreateNewSession}
            className="h-8 text-xs gap-1.5"
          >
            <Plus className="h-3.5 w-3.5" />
            {isEn ? "New Chat" : "Nuova Chat"}
          </Button>

          {chatHistory.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearCurrentSession}
              className="h-8 text-xs text-muted-foreground hover:text-foreground"
            >
              {isEn ? "Clear" : "Pulisci"}
            </Button>
          )}

          {sessions.length > 1 && activeSession && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => handleDeleteSession(activeSession.id)}
              className="h-8 w-8 text-muted-foreground hover:text-destructive"
              title={isEn ? "Delete session" : "Elimina sessione"}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>

        {liveTokPerSec > 0 && (
          <div className="hidden sm:flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
            <span>{liveTokPerSec.toFixed(1)} tok/s</span>
            {lastTelemetry?.rig_name && (
              <span className="text-border">|</span>
            )}
            {lastTelemetry?.rig_name && (
              <span>{lastTelemetry.rig_name}</span>
            )}
          </div>
        )}
      </div>

      {/* Centered Clean Chat Stream */}
      <ScrollArea className="flex-1 py-4 pr-2">
        {chatHistory.length === 0 ? (
          <div className="h-[calc(100vh-21rem)] min-h-[300px] flex flex-col items-center justify-center text-center space-y-4">
            <CutmeshLogo size={44} />
            <div className="space-y-1">
              <h2 className="text-xl font-semibold tracking-tight text-foreground">
                {isEn ? "How can I help you?" : "Come posso aiutarti?"}
              </h2>
              <p className="text-xs text-muted-foreground max-w-sm">
                {isEn
                  ? "Pick a model below and start typing."
                  : "Seleziona il modello qui sotto e inizia a scrivere."}
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {chatHistory.map((m, idx) => {
              const hasThinking = Boolean(m.reasoningContent && m.reasoningContent.length > 0);
              const isThinkingExpanded = expandedThinkingMap[idx] ?? false;
              const isTruncatedByLength = m.finishReason === "length" && !m.isStreaming;

              return (
                <Card
                  key={idx}
                  className={
                    m.role === "user"
                      ? "ml-12 bg-secondary/60 border-border"
                      : "mr-4 bg-card border-border"
                  }
                >
                  <CardHeader className="p-3 pb-1.5">
                    <div className="flex items-center justify-between gap-2 flex-wrap text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <Avatar className="h-5 w-5 border rounded-[2px]">
                          <AvatarFallback className="text-[9px] bg-muted rounded-[2px]">
                            {m.role === "user" ? (
                              <User className="h-3 w-3" />
                            ) : (
                              <Bot className="h-3 w-3 text-primary" />
                            )}
                          </AvatarFallback>
                        </Avatar>
                        <span>{m.role === "user" ? (isEn ? "You" : "Tu") : "Cutmesh"}</span>
                        {m.isStreaming && (
                          <span className="inline-flex items-center gap-1 font-mono lowercase text-[10px] text-primary">
                            <Loader2 className="h-2.5 w-2.5 animate-spin" />
                            {m.isThinking
                              ? isEn
                                ? "thinking..."
                                : "ragionamento..."
                              : isEn
                              ? "writing..."
                              : "scrittura..."}
                          </span>
                        )}
                      </div>

                      {m.role === "assistant" && (
                        <div className="flex items-center gap-2">
                          {typeof m.tokPerSec === "number" && m.tokPerSec > 0 && (
                            <span className="font-mono text-[10px] text-muted-foreground font-normal normal-case">
                              {m.tokPerSec.toFixed(1)} tok/s
                            </span>
                          )}
                          {m.content && (
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              onClick={() => handleCopyMessage(idx, m.content)}
                              className="h-6 px-2 text-[10px] gap-1 font-normal normal-case"
                            >
                              {copiedMsgIdx === idx ? (
                                <>
                                  <Check className="h-3 w-3 text-emerald-500" />
                                  <span className="text-emerald-500">{isEn ? "Copied" : "Copiato"}</span>
                                </>
                              ) : (
                                <>
                                  <Copy className="h-3 w-3" />
                                  <span>{isEn ? "Copy" : "Copia"}</span>
                                </>
                              )}
                            </Button>
                          )}
                        </div>
                      )}
                    </div>
                  </CardHeader>

                  <CardContent className="p-3 pt-1 space-y-2 text-xs">
                    {(hasThinking || (m.isStreaming && m.isThinking)) && (
                      <Collapsible
                        open={isThinkingExpanded}
                        onOpenChange={(open) =>
                          setExpandedThinkingMap((prev) => ({ ...prev, [idx]: open }))
                        }
                        className="border border-border bg-muted/25 overflow-hidden rounded-[2px]"
                      >
                        <CollapsibleTrigger asChild>
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="w-full h-8 flex items-center justify-between px-2.5 text-[11px] font-medium text-muted-foreground hover:text-foreground"
                          >
                            <span>
                              {m.isStreaming && m.isThinking
                                ? isEn
                                  ? "Thinking..."
                                  : "Ragionamento in corso..."
                                : `${isEn ? "Reasoning" : "Ragionamento"}${m.thinkingTokens ? ` (${m.thinkingTokens} ${isEn ? "tokens" : "token"})` : ""}`}
                            </span>
                            {isThinkingExpanded ? (
                              <ChevronDown className="h-3.5 w-3.5" />
                            ) : (
                              <ChevronRight className="h-3.5 w-3.5" />
                            )}
                          </Button>
                        </CollapsibleTrigger>
                        <CollapsibleContent>
                          <Separator />
                          <ScrollArea className="max-h-60 px-3 py-2 text-[11px] text-muted-foreground">
                            <ChatMarkdown
                              content={
                                m.reasoningContent ||
                                (isEn ? "Processing..." : "Elaborazione in corso...")
                              }
                              isStreaming={Boolean(m.isStreaming && m.isThinking)}
                            />
                          </ScrollArea>
                        </CollapsibleContent>
                      </Collapsible>
                    )}

                    {m.role === "user" ? (
                      <div className="whitespace-pre-wrap leading-relaxed text-sm">
                        {m.content}
                      </div>
                    ) : (
                      <ChatMarkdown
                        content={m.content}
                        isStreaming={Boolean(m.isStreaming && !m.isThinking)}
                      />
                    )}

                    {isTruncatedByLength && idx === chatHistory.length - 1 && (
                      <Alert className="mt-2 py-2 px-3 border-amber-500/40 bg-amber-500/10">
                        <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />
                        <AlertDescription className="flex flex-wrap items-center justify-between gap-2 text-xs">
                          <span className="text-[11px] text-amber-500 font-medium">
                            {isEn
                              ? "Response stopped at max length."
                              : "Risposta interrotta al limite di lunghezza."}
                          </span>
                          <Button
                            variant="outline"
                            size="sm"
                            disabled={sendingChat}
                            onClick={() =>
                              handleSendPlaygroundMessage(
                                isEn
                                  ? "Continue exactly from where you left off without repeating."
                                  : "Continua esattamente da dove ti sei interrotto, senza ripetere il testo precedente."
                              )
                            }
                            className="h-7 text-xs gap-1"
                          >
                            <CornerDownRight className="h-3 w-3" />
                            {isEn ? "Continue" : "Continua"}
                          </Button>
                        </AlertDescription>
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              );
            })}
            <div ref={messagesEndRef} />
          </div>
        )}
      </ScrollArea>

      {/* Clean Minimal Composer (Reference-Style with Inline Model Picker) */}
      <div className="pt-2">
        <div className="border border-border bg-card rounded-[3px] focus-within:border-primary/70 transition-colors">
          <Textarea
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSendPlaygroundMessage();
              }
            }}
            placeholder={isEn ? "Ask anything..." : "Chiedi qualsiasi cosa..."}
            disabled={sendingChat}
            rows={2}
            className="min-h-[56px] max-h-40 resize-none border-0 bg-transparent shadow-none focus-visible:ring-0 px-3.5 pt-3 pb-2 text-sm placeholder:text-muted-foreground/70"
          />

          <div className="flex items-center justify-between gap-2 px-2.5 pb-2.5 pt-1">
            <div className="flex items-center gap-1.5 flex-wrap">
              <Select value={playgroundModel} onValueChange={setPlaygroundModel}>
                <SelectTrigger className="h-7 w-auto min-w-[170px] border-0 bg-transparent hover:bg-secondary/70 px-2 text-xs font-medium text-foreground shadow-none focus:ring-0">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {models.map((m) => (
                    <SelectItem key={m.modelId} value={m.modelId} className="text-xs">
                      {m.displayName}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={thinkingMode}
                onValueChange={(v) =>
                  setThinkingMode(v as "off" | "low" | "medium" | "xhigh")
                }
              >
                <SelectTrigger className="h-7 w-auto border-0 bg-transparent hover:bg-secondary/70 px-2 text-xs text-muted-foreground hover:text-foreground shadow-none focus:ring-0">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="off" className="text-xs">{isEn ? "Direct" : "Diretto"}</SelectItem>
                  <SelectItem value="low" className="text-xs">{isEn ? "Fast" : "Veloce"}</SelectItem>
                  <SelectItem value="medium" className="text-xs">{isEn ? "Balanced" : "Bilanciato"}</SelectItem>
                  <SelectItem value="xhigh" className="text-xs">{isEn ? "Deep" : "Approfondito"}</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {sendingChat ? (
              <Button
                variant="destructive"
                size="sm"
                onClick={handleStopGeneration}
                className="h-8 px-3 gap-1.5 text-xs"
              >
                <Square className="h-3 w-3" />
                Stop
              </Button>
            ) : (
              <Button
                size="sm"
                onClick={() => handleSendPlaygroundMessage()}
                disabled={!chatInput.trim()}
                className="h-8 px-3.5 gap-1.5 text-xs"
              >
                <Send className="h-3.5 w-3.5" />
                {isEn ? "Send" : "Invia"}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
