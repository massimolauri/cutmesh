"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Eye,
  Layers,
  Play,
  Plus,
  PowerOff,
  RefreshCw,
  Search,
  Server,
  Trash2,
  Unplug,
  Wrench,
  X
} from "lucide-react";
import {
  ActiveSection,
  CatalogModel,
  GpuWorkerNode,
  ManagedRig
} from "@/types/control-plane";
import { useI18n } from "@/i18n/translations";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from "@/components/ui/tooltip";

interface ClusterOverviewSectionProps {
  t: Record<string, string>;
  workers: GpuWorkerNode[];
  activeRigs: ManagedRig[];
  models: CatalogModel[];
  onNavigate: (section: ActiveSection) => void;
  onTriggerWatchdog: (workerId: string, gpuName: string) => void;
  onDisconnectWorker: (workerId: string, nodeName: string) => Promise<void>;
  onDeleteWorker: (workerId: string, nodeName: string) => Promise<void>;
  onDissolveRig: (rigId: string) => Promise<void>;
  onCreateOrReconfigureRig: (
    rigName: string,
    modelId: string,
    workerIds: string[]
  ) => Promise<void>;
}

export function ClusterOverviewSection({
  t,
  workers,
  activeRigs,
  models,
  onNavigate,
  onTriggerWatchdog,
  onDisconnectWorker,
  onDeleteWorker,
  onDissolveRig,
  onCreateOrReconfigureRig
}: ClusterOverviewSectionProps) {
  const { locale } = useI18n();
  const isEn = locale === "en";
  const [nowMs, setNowMs] = useState<number>(() => Date.now());
  const [selectedModelId, setSelectedModelId] = useState<string>(
    models[0]?.modelId || "qwen3.8-27b"
  );

  useEffect(() => {
    if (models.length > 0 && !models.some((m) => m.modelId === selectedModelId)) {
      setSelectedModelId(models[0].modelId);
    }
  }, [models, selectedModelId]);
  const [selectedWorkerIds, setSelectedWorkerIds] = useState<string[]>([]);
  const [customRigName, setCustomRigName] = useState<string>("");
  const [reconfiguring, setReconfiguring] = useState<boolean>(false);

  // Scalable 10,000+ Rig View State (Search, Filter, Sort, Pagination, On-Demand Inspector)
  const [rigSearch, setRigSearch] = useState<string>("");
  const [rigModelFilter, setRigModelFilter] = useState<string>("ALL");
  const [rigStatusFilter, setRigStatusFilter] = useState<"ALL" | "READY" | "STARTING">("ALL");
  const [rigSortBy, setRigSortBy] = useState<"name" | "inflight" | "decode" | "vram" | "latency">("name");
  const [rigPage, setRigPage] = useState<number>(1);
  const [rigPageSize, setRigPageSize] = useState<number>(25);
  const [inspectedRigId, setInspectedRigId] = useState<string | null>(null);
  const hasAutoExpandedRigRef = React.useRef<boolean>(false);

  // Scalable Worker Fleet State (Search, Status Filter, Pagination)
  const [workerSearch, setWorkerSearch] = useState<string>("");
  const [workerStatusFilter, setWorkerStatusFilter] = useState<"ALL" | "ONLINE" | "STANDBY" | "OFFLINE">("ALL");
  const [workerPage, setWorkerPage] = useState<number>(1);
  const [workerPageSize, setWorkerPageSize] = useState<number>(10);

  // Auto-open pipeline inspector only once on initial load when there is a single active rig;
  // never re-open after the user explicitly clicks "Nascondi".
  useEffect(() => {
    if (!hasAutoExpandedRigRef.current && activeRigs.length === 1) {
      hasAutoExpandedRigRef.current = true;
      setInspectedRigId(activeRigs[0].id);
    } else if (inspectedRigId && !activeRigs.some((r) => r.id === inspectedRigId)) {
      setInspectedRigId(null);
    }
  }, [activeRigs, inspectedRigId]);

  const filteredRigs = useMemo(() => {
    const q = rigSearch.trim().toLowerCase();
    const list = activeRigs.filter((rig) => {
      if (rigModelFilter !== "ALL" && rig.modelId !== rigModelFilter) return false;
      if (rigStatusFilter === "READY" && !rig.inferenceEngineReady) return false;
      if (rigStatusFilter === "STARTING" && rig.inferenceEngineReady) return false;
      if (q) {
        const matchRig =
          rig.rigName.toLowerCase().includes(q) ||
          rig.modelDisplayName.toLowerCase().includes(q) ||
          rig.modelId.toLowerCase().includes(q) ||
          rig.allocations.some((a) => a.gpuName.toLowerCase().includes(q));
        if (!matchRig) return false;
      }
      return true;
    });
    return [...list].sort((a, b) => {
      switch (rigSortBy) {
        case "inflight":
          return (b.activeInFlightRequests ?? 0) - (a.activeInFlightRequests ?? 0);
        case "decode":
          return (b.lastDecodeTokPerSec ?? 0) - (a.lastDecodeTokPerSec ?? 0);
        case "vram":
          return b.totalAllocatedVramMb - a.totalAllocatedVramMb;
        case "latency":
          return (a.avgLatencyMs ?? 150) - (b.avgLatencyMs ?? 150);
        default:
          return a.rigName.localeCompare(b.rigName);
      }
    });
  }, [activeRigs, rigSearch, rigModelFilter, rigStatusFilter, rigSortBy]);

  const rigTotalPages = Math.max(1, Math.ceil(filteredRigs.length / rigPageSize));
  const safeRigPage = Math.min(rigPage, rigTotalPages);
  const pagedRigs = useMemo(() => {
    const start = (safeRigPage - 1) * rigPageSize;
    return filteredRigs.slice(start, start + rigPageSize);
  }, [filteredRigs, safeRigPage, rigPageSize]);

  const inspectedRig = useMemo(
    () => activeRigs.find((r) => r.id === inspectedRigId) || null,
    [activeRigs, inspectedRigId]
  );

  const filteredWorkers = useMemo(() => {
    const q = workerSearch.trim().toLowerCase();
    return workers.filter((w) => {
      const isOff = w.assignmentState === "OFFLINE_STALE";
      if (workerStatusFilter === "ONLINE" && isOff) return false;
      if (workerStatusFilter === "STANDBY" && w.assignmentState !== "IDLE_STANDBY_POOL") return false;
      if (workerStatusFilter === "OFFLINE" && !isOff) return false;
      if (q) {
        return (
          w.nodeName.toLowerCase().includes(q) ||
          w.gpuName.toLowerCase().includes(q) ||
          (w.assignedModelId || "").toLowerCase().includes(q)
        );
      }
      return true;
    });
  }, [workers, workerSearch, workerStatusFilter]);

  const workerTotalPages = Math.max(1, Math.ceil(filteredWorkers.length / workerPageSize));
  const safeWorkerPage = Math.min(workerPage, workerTotalPages);
  const pagedWorkers = useMemo(() => {
    const start = (safeWorkerPage - 1) * workerPageSize;
    return filteredWorkers.slice(start, start + workerPageSize);
  }, [filteredWorkers, safeWorkerPage, workerPageSize]);

  const rigModelOptions = useMemo(() => {
    const map = new Map<string, string>();
    for (const r of activeRigs) {
      if (!map.has(r.modelId)) map.set(r.modelId, r.modelDisplayName);
    }
    return Array.from(map.entries());
  }, [activeRigs]);

  const onlineWorkers = useMemo(
    () => workers.filter((w) => w.assignmentState !== "OFFLINE_STALE"),
    [workers]
  );
  const offlineWorkers = useMemo(
    () => workers.filter((w) => w.assignmentState === "OFFLINE_STALE"),
    [workers]
  );

  // Real-time Cluster & Node Diagnostics: explains WHY a rig didn't form or why a node/rig failed
  const clusterDiagnostics = useMemo(() => {
    const issues: { severity: "error" | "warning"; title: string; detail: string }[] = [];

    // 1. Degraded / Failed Rigs with exact last runtime log
    for (const rig of activeRigs) {
      if (rig.runtimePhase === "ERROR_DEGRADED") {
        issues.push({
          severity: "error",
          title: `Errore avvio cluster '${rig.rigName}' (${rig.modelDisplayName})`,
          detail: rig.runtimeLastLog || rig.operationalStatusMessage || "Il motore di inferenza si è arrestato o non risponde su /health."
        });
      }
    }

    // 2. Unhealthy RPC or Offline Workers
    const unhealthyOnline = onlineWorkers.filter((w) => !w.rpcHealthy);
    for (const w of unhealthyOnline) {
      issues.push({
        severity: "error",
        title: `Nodo '${w.nodeName}' (${w.gpuName}): demone RPC non sano`,
        detail: `Il worker è connesso ma ggml-rpc-server non risponde (riavvii watchdog: ${w.watchdogRestarts}). Non può essere assegnato a un Rig finché non torna sano.`
      });
    }
    if (offlineWorkers.length > 0) {
      issues.push({
        severity: "warning",
        title: `${offlineWorkers.length} nodo/i GPU offline (heartbeat scaduto)`,
        detail: `Nodi disconnessi: ${offlineWorkers.slice(0, 5).map((w) => `${w.nodeName} (${w.gpuName})`).join(", ")}${offlineWorkers.length > 5 ? ` e altri ${offlineWorkers.length - 5}` : ""}.`
      });
    }

    // 3. Unserved AUTO_PILOT models — only show when GPUs ARE online (avoid noisy banner on idle empty cluster)
    const standbyHealthy = onlineWorkers.filter(
      (w) => w.assignmentState === "IDLE_STANDBY_POOL" && w.rpcHealthy
    );
    const standbyVramMb = standbyHealthy.reduce((sum, w) => sum + w.maxAllocatableVramMb, 0);
    if (onlineWorkers.length > 0) {
      for (const m of models) {
        if (m.orchestrationMode !== "AUTO_PILOT" || m.minActiveRigs <= 0) continue;
        const servingCount = activeRigs.filter(
          (r) => r.modelId === m.modelId && r.lifecycleState === "SERVING"
        ).length;
        if (servingCount === 0) {
          if (standbyHealthy.length === 0) {
            issues.push({
              severity: "warning",
              title: `Modello Auto-Pilot '${m.displayName}' non avviato: nessuna GPU libera in standby`,
              detail: `Tutte le ${onlineWorkers.length} GPU online sono già occupate o con RPC non sano. Richiesti ${(m.requiredVramMb / 1024).toFixed(1)} GB VRAM.`
            });
          } else if (standbyVramMb < m.requiredVramMb) {
            const missingGb = ((m.requiredVramMb - standbyVramMb) / 1024).toFixed(1);
            issues.push({
              severity: "warning",
              title: `Modello Auto-Pilot '${m.displayName}': VRAM insufficiente per formare il Rig`,
              detail: `Richiesti ${(m.requiredVramMb / 1024).toFixed(1)} GB, disponibili ${(standbyVramMb / 1024).toFixed(1)} GB su ${standbyHealthy.length} GPU in standby (mancano ${missingGb} GB).`
            });
          } else {
            issues.push({
              severity: "warning",
              title: `Modello Auto-Pilot '${m.displayName}' non ancora allocato`,
              detail: `VRAM standby (${(standbyVramMb / 1024).toFixed(1)} GB su ${standbyHealthy.length} GPU) presente ma vincoli di topologia/policy (es. max nodi WAN o Stage-0 VRAM) impediscono la chiusura del Rig.`
            });
          }
        }
      }
    }

    return issues;
  }, [activeRigs, onlineWorkers, offlineWorkers, models]);

  // Tick every 1 second so "last seen X seconds ago" and offline status update immediately
  useEffect(() => {
    const id = setInterval(() => setNowMs(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  // Auto-select online workers when initial list arrives
  useEffect(() => {
    if (selectedWorkerIds.length === 0 && onlineWorkers.length > 0) {
      setSelectedWorkerIds(onlineWorkers.map((w) => w.id));
    } else {
      // Remove any worker that went offline from selection
      setSelectedWorkerIds((prev) =>
        prev.filter((id) => onlineWorkers.some((w) => w.id === id))
      );
    }
  }, [onlineWorkers]);

  const totalClusterVramMb = onlineWorkers.reduce((acc, w) => acc + w.totalVramMb, 0);
  const freeClusterVramMb = onlineWorkers.reduce((acc, w) => acc + w.freeVramMb, 0);
  const totalTflops = onlineWorkers.reduce((acc, w) => acc + (w.effectiveTflops || 0), 0);

  const selectedModel =
    models.find((m) => m.modelId === selectedModelId) || models[0];

  // Live Water-Filling Topology Preview Calculation
  const topologyPreview = useMemo(() => {
    const chosen = onlineWorkers.filter((w) => selectedWorkerIds.includes(w.id));
    if (!selectedModel || chosen.length === 0) return null;

    const totalAllocatableMb = chosen.reduce(
      (sum, w) => sum + w.maxAllocatableVramMb,
      0
    );
    const totalPhysicalMb = chosen.reduce((sum, w) => sum + w.totalVramMb, 0);
    const requiredMb = selectedModel.requiredVramMb || 600;
    const targetAllocatedTotal = Math.min(
      totalAllocatableMb,
      Math.max(requiredMb, Math.round(totalAllocatableMb * 0.85))
    );
    const fullModelKvMb = Math.round(
      ((selectedModel.maxContextLength || 262144) / 1024) *
        Math.max(8, selectedModel.kvVramPer1kTokensMb || 24)
    );
    const baseModelFootprintMb =
      requiredMb > fullModelKvMb + 4096 ? requiredMb - fullModelKvMb : requiredMb;
    const residualForKvMb = Math.max(512, totalAllocatableMb - baseModelFootprintMb);
    const kvPer1k = Math.max(8, selectedModel.kvVramPer1kTokensMb || 24);
    const rawContext = Math.floor((residualForKvMb * 1024) / kvPer1k);
    const unlockedContext = Math.min(
      262144,
      Math.max(4096, Math.floor(rawContext / 1024) * 1024)
    );

    const totalLayers = selectedModel.totalLayers || 24;
    let cursor = 0;

    const stages = chosen.map((w, idx) => {
      const fraction =
        totalAllocatableMb > 0 ? w.maxAllocatableVramMb / totalAllocatableMb : 1 / chosen.length;
      const allocatedMb = Math.max(256, Math.round(fraction * targetAllocatedTotal));
      const layersCount =
        idx === chosen.length - 1
          ? totalLayers - cursor
          : Math.max(1, Math.round(fraction * totalLayers));
      const layerStart = cursor;
      const layerEnd = Math.min(totalLayers, layerStart + layersCount);
      cursor = layerEnd;

      const roleLabel =
        chosen.length === 1
          ? "Nodo Principale"
          : idx === 0
          ? "Ingresso (Stage 0)"
          : idx === chosen.length - 1
          ? `Uscita (Stage ${idx})`
          : `Intermedio (Stage ${idx})`;

      return {
        worker: w,
        stageIndex: idx,
        fractionPct: (fraction * 100).toFixed(1),
        splitGb: (allocatedMb / 1024).toFixed(1),
        allocatedMb,
        layerStart,
        layerEnd,
        roleLabel
      };
    });

    return {
      chosen,
      totalPhysicalMb,
      totalAllocatableMb,
      requiredMb,
      unlockedContext,
      tensorSplitCsv: stages.map((s) => s.splitGb).join(","),
      enoughVram: totalAllocatableMb >= requiredMb,
      stages
    };
  }, [onlineWorkers, selectedWorkerIds, selectedModel]);

  const toggleWorkerSelection = (workerId: string) => {
    setSelectedWorkerIds((prev) =>
      prev.includes(workerId)
        ? prev.filter((id) => id !== workerId)
        : [...prev, workerId]
    );
  };

  const handleApplyTopology = async () => {
    if (!topologyPreview || !topologyPreview.enoughVram || !selectedModel) return;
    setReconfiguring(true);
    const autoName =
      customRigName.trim() ||
      `rig-${selectedModel.modelId.replace("-covariant", "")}-${topologyPreview.chosen.length}gpu`;
    await onCreateOrReconfigureRig(
      autoName,
      selectedModel.modelId,
      topologyPreview.chosen.map((w) => w.id)
    );
    setCustomRigName("");
    setReconfiguring(false);
  };

  const formatHeartbeatAge = (iso?: string) => {
    if (!iso) return "N/D";
    const diffSec = Math.max(0, Math.round((nowMs - new Date(iso).getTime()) / 1000));
    if (diffSec <= 1) return "adesso (1s)";
    if (diffSec < 60) return `${diffSec}s fa`;
    return `${Math.floor(diffSec / 60)}m ${diffSec % 60}s fa`;
  };

  return (
    <div className="space-y-6">
      {/* KPI TELEMETRY CARDS */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center justify-between">
              <span>Nodi GPU Online</span>
              <Server className="w-4 h-4 text-muted-foreground" />
            </CardDescription>
            <CardTitle className="text-2xl flex items-baseline gap-2">
              {onlineWorkers.length}
              <span className="text-xs font-normal text-muted-foreground">
                online {offlineWorkers.length > 0 ? `· ${offlineWorkers.length} disconnessi` : ""}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              Heartbeat rapido ogni 2s (timeout 8s)
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center justify-between">
              <span>VRAM Cluster Aggregata</span>
              <Cpu className="w-4 h-4 text-muted-foreground" />
            </CardDescription>
            <CardTitle className="text-2xl">
              {(totalClusterVramMb / 1024).toFixed(1)} GB
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-1.5">
            <Progress
              value={
                totalClusterVramMb > 0
                  ? ((totalClusterVramMb - freeClusterVramMb) / totalClusterVramMb) * 100
                  : 0
              }
              className="h-1.5"
            />
            <p className="text-xs text-muted-foreground">
              {(freeClusterVramMb / 1024).toFixed(1)}{" "}
              {isEn ? "GB free for context & inference" : "GB liberi per contesto ed elaborazione"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center justify-between">
              <span>{t.kpiComputePower}</span>
              <Activity className="w-4 h-4 text-muted-foreground" />
            </CardDescription>
            <CardTitle className="text-2xl">{totalTflops.toFixed(1)} TFLOPS</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              {isEn
                ? "Automatic layer sharding across online GPUs"
                : "Distribuzione automatica dei layer sulle GPU online"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center justify-between">
              <span>{t.kpiActiveRigs}</span>
              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
            </CardDescription>
            <CardTitle className="text-2xl flex items-baseline gap-2">
              {activeRigs.length}
              <Badge variant="secondary" className="text-[10px]">
                {isEn ? "Operational" : "Operativi"}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              {isEn
                ? "Automatic reconfiguration on node disconnect"
                : "Riconfigurazione automatica in caso di disconnessione"}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* SECTION 1: VISUAL END-TO-END PIPELINE TOPOLOGY (ACTIVE RIGS) */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <CardTitle className="text-base">{t.pipelineTitle}</CardTitle>
              <CardDescription>{t.pipelineSubtitle}</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => onNavigate("provider")}
                className="h-8 text-xs"
              >
                <Plus className="w-3.5 h-3.5 mr-1.5" />
                {t.connectNewGpuBtn}
              </Button>
              <Button
                size="sm"
                onClick={() => onNavigate("playground")}
                className="h-8 text-xs"
              >
                <Play className="w-3.5 h-3.5 mr-1.5" />
                {isEn ? "Open Playground" : "Apri Playground"}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {clusterDiagnostics.length > 0 && (
            <div className="space-y-2">
              {clusterDiagnostics.map((diag, idx) => (
                <Alert
                  key={idx}
                  variant={diag.severity === "error" ? "destructive" : "default"}
                  className={
                    diag.severity === "error"
                      ? "border-destructive/60 bg-destructive/10 py-2.5"
                      : "border-amber-500/50 bg-amber-500/10 py-2.5"
                  }
                >
                  <AlertTriangle
                    className={`h-4 w-4 ${
                      diag.severity === "error" ? "text-destructive" : "text-amber-500"
                    }`}
                  />
                  <AlertTitle className="text-xs font-semibold">
                    {diag.title}
                  </AlertTitle>
                  <AlertDescription className="text-xs text-muted-foreground font-mono">
                    {diag.detail}
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          )}

          {activeRigs.length === 0 ? (
            <Alert className="border-dashed bg-muted/20 py-6">
              <Layers className="h-4 w-4 text-primary" />
              <AlertTitle className="text-sm font-semibold">
                {isEn ? "No active clusters right now" : "Nessun cluster attivo al momento"}
              </AlertTitle>
              <AlertDescription className="text-xs text-muted-foreground">
                {isEn ? (
                  <>
                    Use the <strong>Cluster Composer below</strong> to select online GPUs and start a model.
                  </>
                ) : (
                  <>
                    Usa il <strong>Configuratore qui sotto</strong> per selezionare le GPU online e avviare un modello.
                  </>
                )}
              </AlertDescription>
            </Alert>
          ) : (
            <div className="space-y-4">
              {/* Scalable Search / Filter / Sort / Page-Size Toolbar */}
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border bg-muted/20 p-2.5">
                <div className="flex flex-wrap items-center gap-2 flex-1 min-w-[240px]">
                  <div className="relative flex-1 min-w-[180px] max-w-xs">
                    <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-muted-foreground" />
                    <Input
                      value={rigSearch}
                      onChange={(e) => {
                        setRigSearch(e.target.value);
                        setRigPage(1);
                      }}
                      placeholder={isEn ? "Search rig, model, or GPU..." : "Cerca rig, modello o GPU..."}
                      className="pl-8 h-8 text-xs"
                    />
                  </div>
                  <Select
                    value={rigModelFilter}
                    onValueChange={(v) => {
                      setRigModelFilter(v);
                      setRigPage(1);
                    }}
                  >
                    <SelectTrigger className="h-8 w-[180px] text-xs">
                      <SelectValue placeholder={isEn ? "Model" : "Modello"} />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ALL" className="text-xs">
                        {isEn
                          ? `All models (${rigModelOptions.length})`
                          : `Tutti i modelli (${rigModelOptions.length})`}
                      </SelectItem>
                      {rigModelOptions.map(([id, name]) => (
                        <SelectItem key={id} value={id} className="text-xs">{name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select
                    value={rigStatusFilter}
                    onValueChange={(v: "ALL" | "READY" | "STARTING") => {
                      setRigStatusFilter(v);
                      setRigPage(1);
                    }}
                  >
                    <SelectTrigger className="h-8 w-[145px] text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ALL" className="text-xs">{isEn ? "All states" : "Tutti gli stati"}</SelectItem>
                      <SelectItem value="READY" className="text-xs">{isEn ? "Operational (200 OK)" : "Operativi (200 OK)"}</SelectItem>
                      <SelectItem value="STARTING" className="text-xs">{isEn ? "Starting / Boot" : "In avvio / Boot"}</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select
                    value={rigSortBy}
                    onValueChange={(v: "name" | "inflight" | "decode" | "vram" | "latency") => setRigSortBy(v)}
                  >
                    <SelectTrigger className="h-8 w-[155px] text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="name" className="text-xs">{isEn ? "Sort: Name" : "Ordina: Nome"}</SelectItem>
                      <SelectItem value="inflight" className="text-xs">{isEn ? "Sort: In-Flight" : "Ordina: In-Flight"}</SelectItem>
                      <SelectItem value="decode" className="text-xs">{isEn ? "Sort: Decode tok/s" : "Ordina: Decode tok/s"}</SelectItem>
                      <SelectItem value="vram" className="text-xs">{isEn ? "Sort: VRAM" : "Ordina: VRAM"}</SelectItem>
                      <SelectItem value="latency" className="text-xs">{isEn ? "Sort: Latency" : "Ordina: Latenza"}</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center gap-2 text-xs text-muted-foreground font-mono">
                  <Badge variant="secondary" className="font-mono text-[11px]">
                    {filteredRigs.length.toLocaleString()} / {activeRigs.length.toLocaleString()} rig
                  </Badge>
                  <Select
                    value={String(rigPageSize)}
                    onValueChange={(v) => {
                      setRigPageSize(Number(v));
                      setRigPage(1);
                    }}
                  >
                    <SelectTrigger className="h-8 w-[90px] text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="10" className="text-xs">10 / {isEn ? "pg" : "pag"}</SelectItem>
                      <SelectItem value="25" className="text-xs">25 / {isEn ? "pg" : "pag"}</SelectItem>
                      <SelectItem value="50" className="text-xs">50 / {isEn ? "pg" : "pag"}</SelectItem>
                      <SelectItem value="100" className="text-xs">100 / {isEn ? "pg" : "pag"}</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Compact High-Density Rigs Table (handles 10,000+ rigs at 60fps) */}
              <div className="rounded-lg border overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow className="bg-muted/30">
                      <TableHead className="text-xs h-9">{isEn ? "Status" : "Stato"}</TableHead>
                      <TableHead className="text-xs h-9">Cluster / Rig</TableHead>
                      <TableHead className="text-xs h-9">{isEn ? "Model" : "Modello"}</TableHead>
                      <TableHead className="text-xs h-9">GPU / VRAM</TableHead>
                      <TableHead className="text-xs h-9">{isEn ? "Context" : "Contesto"}</TableHead>
                      <TableHead className="text-xs h-9">Prefill / Decode</TableHead>
                      <TableHead className="text-xs h-9">{isEn ? "Load" : "Carico"}</TableHead>
                      <TableHead className="text-xs h-9 text-right">{isEn ? "Actions" : "Azioni"}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {pagedRigs.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={8} className="text-center py-6 text-xs text-muted-foreground">
                          {isEn ? "No rigs match the selected filters." : "Nessun rig corrisponde ai filtri selezionati."}
                        </TableCell>
                      </TableRow>
                    ) : (
                      pagedRigs.map((rig) => {
                        const isInspected = inspectedRigId === rig.id;
                        return (
                          <TableRow
                            key={rig.id}
                            onClick={() => setInspectedRigId(isInspected ? null : rig.id)}
                            className={`cursor-pointer ${isInspected ? "bg-primary/5" : ""}`}
                          >
                            <TableCell className="py-2">
                              {rig.inferenceEngineReady ? (
                                <Badge className="bg-emerald-600 hover:bg-emerald-600 text-white font-mono text-[10px] gap-1">
                                  <span className="h-1.5 w-1.5 rounded-full bg-white animate-pulse" />
                                  OPERATIVO
                                </Badge>
                              ) : rig.runtimePhase === "ERROR_DEGRADED" ? (
                                <Badge variant="destructive" className="font-mono text-[10px] gap-1">
                                  <AlertTriangle className="w-2.5 h-2.5" />
                                  ERRORE
                                </Badge>
                              ) : rig.runtimePhase === "LOADING_VRAM" ? (
                                <Badge className="bg-blue-600 hover:bg-blue-600 text-white font-mono text-[10px] gap-1">
                                  <RefreshCw className="w-2.5 h-2.5 animate-spin" />
                                  FASE 3/4
                                </Badge>
                              ) : rig.runtimePhase === "PREWARMING_SHARDS" ? (
                                <Badge className="bg-indigo-600 hover:bg-indigo-600 text-white font-mono text-[10px] gap-1">
                                  <RefreshCw className="w-2.5 h-2.5 animate-spin" />
                                  FASE 2/4
                                </Badge>
                              ) : (
                                <Badge variant="outline" className="border-amber-500/60 text-amber-400 font-mono text-[10px]">
                                  FASE 1/4
                                </Badge>
                              )}
                            </TableCell>
                            <TableCell className="py-2 font-semibold text-xs font-mono">
                              {rig.rigName}
                            </TableCell>
                            <TableCell className="py-2 text-xs">
                              <Badge variant="secondary" className="text-[11px] font-normal">
                                {rig.modelDisplayName}
                              </Badge>
                            </TableCell>
                            <TableCell className="py-2 font-mono text-xs">
                              {rig.allocations.length} GPU · {(rig.totalAllocatedVramMb / 1024).toFixed(1)} GB
                            </TableCell>
                            <TableCell className="py-2 font-mono text-xs text-muted-foreground">
                              {rig.autoMaximizedContextSize.toLocaleString()} tok
                            </TableCell>
                            <TableCell className="py-2 font-mono text-xs">
                              {(rig.lastPrefillTokPerSec && rig.lastPrefillTokPerSec > 0) ||
                              (rig.lastDecodeTokPerSec && rig.lastDecodeTokPerSec > 0) ? (
                                <span>
                                  <span className="text-emerald-500 font-semibold">
                                    {rig.lastPrefillTokPerSec && rig.lastPrefillTokPerSec > 0
                                      ? rig.lastPrefillTokPerSec.toFixed(1)
                                      : "—"}
                                  </span>
                                  {" / "}
                                  <span className="text-sky-400 font-semibold">
                                    {rig.lastDecodeTokPerSec && rig.lastDecodeTokPerSec > 0
                                      ? rig.lastDecodeTokPerSec.toFixed(1)
                                      : "—"}
                                  </span>
                                  <span className="text-[10px] text-muted-foreground"> tok/s</span>
                                </span>
                              ) : (
                                <span className="text-muted-foreground text-[11px]">—</span>
                              )}
                            </TableCell>
                            <TableCell className="py-2 font-mono text-xs text-muted-foreground">
                              {rig.activeInFlightRequests ?? 0} in-flight · {rig.servedRequestsCount} req
                              {rig.servedRequestsCount > 0 && rig.avgLatencyMs && rig.avgLatencyMs > 0
                                ? ` · ${Math.round(rig.avgLatencyMs)} ms`
                                : ""}
                            </TableCell>
                            <TableCell className="py-2 text-right" onClick={(e) => e.stopPropagation()}>
                              <div className="inline-flex items-center gap-1.5">
                                <Button
                                  variant={isInspected ? "default" : "outline"}
                                  size="sm"
                                  onClick={() => setInspectedRigId(isInspected ? null : rig.id)}
                                  className="h-7 text-[11px] px-2"
                                >
                                  <Eye className="w-3 h-3 mr-1" />
                                  {isInspected ? "Nascondi" : "Topologia"}
                                </Button>
                                <Button
                                  variant="destructive"
                                  size="sm"
                                  onClick={() => onDissolveRig(rig.id)}
                                  className="h-7 text-[11px] px-2"
                                >
                                  <PowerOff className="w-3 h-3" />
                                </Button>
                              </div>
                            </TableCell>
                          </TableRow>
                        );
                      })
                    )}
                  </TableBody>
                </Table>
              </div>

              {/* Pagination Controls */}
              {rigTotalPages > 1 && (
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span className="font-mono">
                    Pagina {safeRigPage} di {rigTotalPages} ({filteredRigs.length.toLocaleString()} rig totali)
                  </span>
                  <div className="flex items-center gap-1.5">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={safeRigPage <= 1}
                      onClick={() => setRigPage((p) => Math.max(1, p - 1))}
                      className="h-7 text-xs px-2"
                    >
                      <ChevronLeft className="w-3.5 h-3.5 mr-1" />
                      Precedente
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={safeRigPage >= rigTotalPages}
                      onClick={() => setRigPage((p) => Math.min(rigTotalPages, p + 1))}
                      className="h-7 text-xs px-2"
                    >
                      Successiva
                      <ChevronRight className="w-3.5 h-3.5 ml-1" />
                    </Button>
                  </div>
                </div>
              )}

              {/* On-Demand Detailed Pipeline Topology Inspector (1 rig at a time) */}
              {(inspectedRig ? [inspectedRig] : []).map((rig) => {
              const totalLayers =
                rig.allocations.length > 0
                  ? Math.max(...rig.allocations.map((a) => a.layerEnd), 24)
                  : 24;

              const rigPrefillTokPerSec =
                rig.lastPrefillTokPerSec && rig.lastPrefillTokPerSec > 0
                  ? Number(rig.lastPrefillTokPerSec.toFixed(1))
                  : null;
              const rigDecodeTokPerSec =
                rig.lastDecodeTokPerSec && rig.lastDecodeTokPerSec > 0
                  ? Number(rig.lastDecodeTokPerSec.toFixed(1))
                  : null;
              const hasLiveMeasuredSpeed =
                rigPrefillTokPerSec !== null || rigDecodeTokPerSec !== null;

              return (
                <Card key={rig.id} className="shadow-sm">
                  {/* Rig Top Control Bar */}
                  <CardHeader className="p-4 pb-3 border-b">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="flex flex-wrap items-center gap-2">
                        {rig.inferenceEngineReady ? (
                          <Badge className="bg-emerald-600 hover:bg-emerald-600 text-white font-mono text-[10px] gap-1.5">
                            <span className="h-1.5 w-1.5 rounded-full bg-white animate-pulse" />
                            OPERATIVO (200 OK)
                          </Badge>
                        ) : rig.runtimePhase === "LOADING_VRAM" ? (
                          <Badge className="bg-blue-600 hover:bg-blue-600 text-white font-mono text-[10px] gap-1.5">
                            <RefreshCw className="w-3 h-3 animate-spin" />
                            FASE 3/4 · CARICAMENTO IN VRAM...
                          </Badge>
                        ) : rig.runtimePhase === "PREWARMING_SHARDS" ? (
                          <Badge className="bg-indigo-600 hover:bg-indigo-600 text-white font-mono text-[10px] gap-1.5">
                            <RefreshCw className="w-3 h-3 animate-spin" />
                            FASE 2/4 · PREPARAZIONE PESI...
                          </Badge>
                        ) : (
                          <Badge
                            variant="outline"
                            className="border-amber-500/60 bg-amber-500/10 text-amber-400 font-mono text-[10px] gap-1.5"
                          >
                            <AlertTriangle className="w-3 h-3 text-amber-400" />
                            FASE 1/4 · ALLOCAZIONE LAYER
                          </Badge>
                        )}
                        <span className="font-semibold text-sm">{rig.rigName}</span>
                        <Badge variant="secondary" className="text-xs">
                          {rig.modelDisplayName}
                        </Badge>
                        <Badge variant="outline" className="font-mono text-[11px]">
                          Distribuzione {rig.tensorSplitCsv}
                        </Badge>
                        <Badge variant="outline" className="font-mono text-[11px]">
                          Contesto {rig.autoMaximizedContextSize.toLocaleString()} tok
                        </Badge>
                        {hasLiveMeasuredSpeed && (
                          <Badge
                            variant="outline"
                            className="font-mono text-[11px] border-emerald-500/50 bg-emerald-500/10 text-emerald-400"
                          >
                            {`Prefill: ${rigPrefillTokPerSec ?? "—"} tok/s · Decode: ${rigDecodeTokPerSec ?? "—"} tok/s`}
                          </Badge>
                        )}
                        <Badge variant="outline" className="font-mono text-[11px]">
                          Richieste attive: {rig.activeInFlightRequests ?? 0}
                          {rig.servedRequestsCount > 0 && rig.avgLatencyMs && rig.avgLatencyMs > 0
                            ? ` · ${Math.round(rig.avgLatencyMs)} ms`
                            : ""}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary" className="font-mono text-[11px] font-normal">
                          {(rig.totalAllocatedVramMb / 1024).toFixed(1)} GB VRAM ·{" "}
                          {rig.servedRequestsCount} req servite
                        </Badge>
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={() => onDissolveRig(rig.id)}
                          className="h-7 text-xs px-2.5"
                        >
                          <PowerOff className="w-3 h-3 mr-1" />
                          Arresta Cluster
                        </Button>
                      </div>
                    </div>
                  </CardHeader>

                  <CardContent className="p-4 space-y-4">
                    {/* FINITE STATE MACHINE (FSM) 4-STEP LIFECYCLE BAR */}
                    {(() => {
                      const currentStep = rig.inferenceEngineReady
                        ? 4
                        : rig.runtimeStep ||
                          (rig.runtimePhase === "LOADING_VRAM"
                            ? 3
                            : rig.runtimePhase === "PREWARMING_SHARDS"
                            ? 2
                            : 1);
                      const fsmSteps = [
                        {
                          step: 1,
                          title: "1. Layer Allocati",
                          subtitle: `${rig.allocations.length} GPU · 0→${totalLayers} layer`
                        },
                        {
                          step: 2,
                          title: "2. Preparazione Pesi",
                          subtitle: "Sincronizzazione shard sui nodi"
                        },
                        {
                          step: 3,
                          title: "3. Caricamento Motore",
                          subtitle: "Mappatura memoria VRAM e collegamenti"
                        },
                        {
                          step: 4,
                          title: "4. Operativo",
                          subtitle: "Pronto per richieste /v1"
                        }
                      ];

                      return (
                        <Card className="bg-muted/25 shadow-none">
                          <CardContent className="p-3 space-y-2.5">
                            <div className="grid grid-cols-1 sm:grid-cols-4 gap-2">
                              {fsmSteps.map((s) => {
                                const isDone =
                                  currentStep > s.step || (s.step === 4 && rig.inferenceEngineReady);
                                const isCurrent = currentStep === s.step && !rig.inferenceEngineReady;
                                return (
                                  <Card
                                    key={s.step}
                                    className={`shadow-none transition-all ${
                                      isDone
                                        ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-400"
                                        : isCurrent
                                        ? "border-primary bg-primary/10 text-foreground ring-1 ring-primary/40"
                                        : "border-border/60 bg-background/40 text-muted-foreground opacity-60"
                                    }`}
                                  >
                                    <CardContent className="px-3 py-2 flex items-center gap-2.5">
                                      <div className="shrink-0">
                                        {isDone ? (
                                          <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                                        ) : isCurrent ? (
                                          <RefreshCw className="w-4 h-4 text-primary animate-spin" />
                                        ) : (
                                          <Badge
                                            variant="outline"
                                            className="h-5 w-5 p-0 rounded-full flex items-center justify-center text-[10px] font-mono"
                                          >
                                            {s.step}
                                          </Badge>
                                        )}
                                      </div>
                                      <div className="min-w-0">
                                        <div className="text-xs font-semibold truncate">{s.title}</div>
                                        <div className="text-[10px] font-mono text-muted-foreground truncate">
                                          {s.subtitle}
                                        </div>
                                      </div>
                                    </CardContent>
                                  </Card>
                                );
                              })}
                            </div>
                            <Separator className="bg-border/50" />
                            <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-muted-foreground">
                              <span>
                                Stato Macchina a Stati:{" "}
                                <strong className="text-foreground">
                                  {rig.operationalStatusMessage || "Fase 1/4 · Layer Allocati"}
                                </strong>
                              </span>
                              {rig.runtimeLastLog && (
                                <Badge variant="outline" className="truncate max-w-xl text-[10px] font-mono">
                                  Stato avvio: {rig.runtimeLastLog}
                                </Badge>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      );
                    })()}

                    {/* VISUAL HORIZONTAL PIPELINE FLOW */}
                    <div className="flex flex-col lg:flex-row items-stretch gap-3 overflow-x-auto pb-1">
                      {/* INGRESS GATEWAY NODE */}
                      <Card className="lg:w-48 shrink-0 bg-muted/30 flex flex-col justify-between shadow-none">
                        <CardContent className="p-3 space-y-1">
                          <Badge variant="outline" className="text-[10px] font-mono">
                            INGRESSO API
                          </Badge>
                          <div className="text-xs font-semibold pt-1">OpenAI `/v1/chat`</div>
                          <p className="text-[11px] text-muted-foreground">
                            Instradamento richieste verso il primo nodo del cluster.
                          </p>
                        </CardContent>
                        <CardFooter className="p-3 pt-2 border-t flex items-center justify-between text-[10px] font-mono text-muted-foreground">
                          <span>Gateway</span>
                          <Badge variant="secondary" className="text-[9px] text-emerald-500">
                            ATTIVO
                          </Badge>
                        </CardFooter>
                      </Card>

                      {/* GPU STAGES IN PIPELINE */}
                      {rig.allocations.map((alloc) => {
                        const liveWorker = workers.find((w) => w.id === alloc.workerNodeId);
                        const isOnline =
                          liveWorker &&
                          liveWorker.assignmentState !== "OFFLINE_STALE" &&
                          liveWorker.rpcHealthy;
                        const splitPct = Math.round(alloc.tensorSplitFraction * 100);
                        const liveUsedVramMb = liveWorker
                          ? Math.max(0, liveWorker.totalVramMb - liveWorker.freeVramMb)
                          : alloc.allocatedVramMb;
                        const totalNodeVramMb = liveWorker?.totalVramMb || alloc.allocatedVramMb;
                        const vramRatio = Math.min(
                          100,
                          Math.round((liveUsedVramMb / Math.max(1, totalNodeVramMb)) * 100)
                        );

                        return (
                          <React.Fragment key={alloc.id}>
                            {/* Animated Connector Arrow */}
                            <div className="flex lg:flex-col items-center justify-center px-1 text-muted-foreground shrink-0">
                              <ArrowRight className="w-4 h-4 text-primary hidden lg:block" />
                              <Badge variant="outline" className="text-[9px] font-mono px-1.5 py-0">
                                P2P
                              </Badge>
                            </div>

                            {/* Stage GPU Card */}
                            <Card
                              className={`flex-1 min-w-[265px] flex flex-col justify-between shadow-none transition-colors ${
                                isOnline
                                  ? "bg-background border-border"
                                  : "bg-destructive/5 border-destructive"
                              }`}
                            >
                              <CardContent className="p-3.5 space-y-2.5">
                                <div className="flex items-center justify-between gap-2">
                                  <Badge variant="secondary" className="font-mono text-[10px]">
                                    STAGE #{alloc.pipelineStageIndex} · {splitPct}% SPLIT
                                  </Badge>
                                  <div className="flex items-center gap-1.5 text-[11px]">
                                    <span
                                      className={`h-2 w-2 rounded-full ${
                                        isOnline
                                          ? "bg-emerald-500 animate-pulse"
                                          : "bg-destructive"
                                      }`}
                                    />
                                    <span className="font-mono text-muted-foreground">
                                      {liveWorker
                                        ? formatHeartbeatAge(liveWorker.lastHeartbeatAt)
                                        : "OFFLINE"}
                                    </span>
                                  </div>
                                </div>

                                <div>
                                  <div className="text-sm font-semibold flex items-center justify-between">
                                    <span>{alloc.gpuName}</span>
                                  </div>
                                  <div className="text-xs font-mono text-muted-foreground">
                                    Server: {liveWorker?.nodeName || alloc.irohNodeId.slice(0, 12)}
                                  </div>
                                </div>

                                {/* Layer Sharding Visual Bar */}
                                <div className="space-y-1 pt-1">
                                  <div className="flex justify-between text-[11px]">
                                    <span className="font-medium">
                                      Layer Modello {alloc.layerStart} → {alloc.layerEnd}
                                    </span>
                                    <span className="font-mono text-muted-foreground">
                                      {alloc.layerEnd - alloc.layerStart} / {totalLayers} layer
                                    </span>
                                  </div>
                                  <Progress
                                    value={Math.min(
                                      100,
                                      Math.max(
                                        8,
                                        ((alloc.layerEnd - alloc.layerStart) / totalLayers) * 100
                                      )
                                    )}
                                    className="h-2"
                                  />
                                </div>

                                {/* Real Live VRAM Usage Bar */}
                                <div className="space-y-1">
                                  <div className="flex justify-between text-[11px]">
                                    <span className="text-muted-foreground">
                                      {isEn ? "VRAM Used (Live)" : "VRAM Usata (Live)"}
                                    </span>
                                    <span className="font-mono">
                                      {liveUsedVramMb.toLocaleString()} /{" "}
                                      {totalNodeVramMb.toLocaleString()} MB ({vramRatio}%)
                                    </span>
                                  </div>
                                  <Progress value={vramRatio} className="h-1.5" />
                                </div>

                                {/* Real measured Rig Prefill & Decode telemetry (only shown after real traffic) */}
                                {hasLiveMeasuredSpeed && (
                                  <div className="flex items-center justify-between text-[10px] font-mono pt-1 border-t border-border/50 text-muted-foreground">
                                    <span>Throughput Rig:</span>
                                    <span className="text-foreground font-semibold">
                                      <span className="text-emerald-500">{rigPrefillTokPerSec ?? "—"}</span> prefill ·{" "}
                                      <span className="text-sky-400">{rigDecodeTokPerSec ?? "—"}</span> decode tok/s
                                    </span>
                                  </div>
                                )}
                              </CardContent>

                              <CardFooter className="p-3.5 pt-3 border-t flex items-center justify-between gap-2">
                                <div className="text-[11px] font-mono text-muted-foreground">
                                  {liveWorker
                                    ? `${liveWorker.temperatureC}°C · ${liveWorker.effectiveBandwidthGbS} GB/s`
                                    : "Disconnesso"}
                                </div>
                                {liveWorker && (
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() =>
                                      onDisconnectWorker(liveWorker.id, liveWorker.nodeName)
                                    }
                                    className="h-7 text-[11px] px-2 text-destructive hover:text-destructive"
                                  >
                                    <Unplug className="w-3 h-3 mr-1" />
                                    Scollega GPU
                                  </Button>
                                )}
                              </CardFooter>
                            </Card>
                          </React.Fragment>
                        );
                      })}

                      {/* EGRESS NODE */}
                      <div className="flex lg:flex-col items-center justify-center px-1 text-muted-foreground shrink-0">
                        <ArrowRight className="w-4 h-4 text-primary hidden lg:block" />
                        <Badge variant="outline" className="text-[9px] font-mono px-1.5 py-0">
                          SSE
                        </Badge>
                      </div>

                      <Card className="lg:w-48 shrink-0 bg-muted/30 flex flex-col justify-between shadow-none">
                        <CardContent className="p-3 space-y-1">
                          <Badge variant="outline" className="text-[10px] font-mono">
                            USCITA RISPOSTA
                          </Badge>
                          <div className="text-xs font-semibold pt-1">
                            Streaming SSE
                          </div>
                          <p className="text-[11px] text-muted-foreground">
                            Consegna dei token generati verso il client API.
                          </p>
                        </CardContent>
                        <CardFooter className="p-3 pt-2 border-t flex items-center justify-between text-[10px] font-mono">
                          <span>Stato Stream</span>
                          <Badge variant="secondary" className="text-[9px] text-emerald-500">
                            ATTIVO
                          </Badge>
                        </CardFooter>
                      </Card>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* SECTION 2: INTERACTIVE TOPOLOGY COMPOSER & LIVE NODE POOL */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Interactive GPU Node Selector & Fleet Status */}
        <Card className="lg:col-span-7">
          <CardHeader className="pb-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <CardTitle className="text-base">
                  Flotta Nodi GPU (Clicca le GPU per comporre il Cluster)
                </CardTitle>
                <CardDescription>
                  Seleziona una o più GPU online per distribuire i layer del modello o scollega i nodi.
                </CardDescription>
              </div>
              <Badge variant="outline" className="font-mono text-xs">
                {selectedWorkerIds.length} / {workers.length} GPU selezionate
              </Badge>
            </div>
            {workers.length > 0 && (
              <div className="flex flex-wrap items-center gap-2 pt-2">
                <div className="relative flex-1 min-w-[160px]">
                  <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-muted-foreground" />
                  <Input
                    value={workerSearch}
                    onChange={(e) => {
                      setWorkerSearch(e.target.value);
                      setWorkerPage(1);
                    }}
                    placeholder="Filtra nodo o GPU..."
                    className="pl-8 h-8 text-xs"
                  />
                </div>
                <Select
                  value={workerStatusFilter}
                  onValueChange={(v: "ALL" | "ONLINE" | "STANDBY" | "OFFLINE") => {
                    setWorkerStatusFilter(v);
                    setWorkerPage(1);
                  }}
                >
                  <SelectTrigger className="h-8 w-[140px] text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ALL" className="text-xs">Tutte ({workers.length})</SelectItem>
                    <SelectItem value="ONLINE" className="text-xs">Online ({onlineWorkers.length})</SelectItem>
                    <SelectItem value="STANDBY" className="text-xs">Standby</SelectItem>
                    <SelectItem value="OFFLINE" className="text-xs">Offline ({offlineWorkers.length})</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}
          </CardHeader>
          <CardContent className="space-y-3">
            {workers.length === 0 ? (
              <Alert className="border-dashed bg-muted/20">
                <Server className="h-4 w-4" />
                <AlertTitle className="text-sm font-medium">
                  Nessun nodo GPU registrato
                </AlertTitle>
                <AlertDescription className="text-xs text-muted-foreground">
                  Collega un server GPU per vederlo apparire qui in tempo reale.
                </AlertDescription>
              </Alert>
            ) : (
              <>
              {pagedWorkers.map((w) => {
                const isOffline = w.assignmentState === "OFFLINE_STALE";
                const isSelected = selectedWorkerIds.includes(w.id);
                const parentRig = activeRigs.find((r) =>
                  r.allocations.some((a) => a.workerNodeId === w.id)
                );
                const realPrefill =
                  parentRig?.lastPrefillTokPerSec && parentRig.lastPrefillTokPerSec > 0
                    ? parentRig.lastPrefillTokPerSec.toFixed(1)
                    : null;
                const realDecode =
                  parentRig?.lastDecodeTokPerSec && parentRig.lastDecodeTokPerSec > 0
                    ? parentRig.lastDecodeTokPerSec.toFixed(1)
                    : null;

                return (
                  <Card
                    key={w.id}
                    onClick={() => {
                      if (!isOffline) toggleWorkerSelection(w.id);
                    }}
                    className={`shadow-none transition-all ${
                      isOffline
                        ? "border-destructive/50 bg-destructive/5 opacity-85"
                        : isSelected
                        ? "border-primary bg-primary/5 ring-1 ring-primary cursor-pointer"
                        : "hover:border-primary/50 cursor-pointer"
                    }`}
                  >
                    <CardContent className="p-3.5">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2.5">
                          <Checkbox
                            checked={isSelected && !isOffline}
                            disabled={isOffline}
                            onClick={(e) => e.stopPropagation()}
                            onCheckedChange={() => {
                              if (!isOffline) toggleWorkerSelection(w.id);
                            }}
                          />
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="font-semibold text-sm">{w.nodeName}</span>
                              <Badge
                                variant={
                                  isOffline
                                    ? "destructive"
                                    : w.assignmentState === "LOCKED_IN_RIG"
                                    ? "default"
                                    : "secondary"
                                }
                                className="text-[10px] font-mono"
                              >
                                {isOffline
                                  ? "DISCONNESSO (OFFLINE)"
                                  : w.assignmentState === "LOCKED_IN_RIG"
                                  ? "ATTIVO"
                                  : "STANDBY DISPONIBILE"}
                              </Badge>
                            </div>
                            <div className="text-xs text-muted-foreground flex flex-wrap items-center gap-2 mt-0.5">
                              <span className="font-medium text-foreground">{w.gpuName}</span>
                              <span>·</span>
                              <span className="font-mono">
                                {w.freeVramMb} / {w.totalVramMb} MB VRAM
                              </span>
                              <span>·</span>
                              <span className="font-mono">
                                {w.effectiveTflops} TFLOPS · {w.effectiveBandwidthGbS} GB/s
                              </span>
                              {realPrefill && (
                                <>
                                  <span>·</span>
                                  <span className="font-mono text-emerald-500 font-semibold">
                                    Prefill: {realPrefill} tok/s
                                  </span>
                                </>
                              )}
                              {realDecode && (
                                <>
                                  <span>·</span>
                                  <span className="font-mono text-sky-400 font-semibold">
                                    Decode: {realDecode} tok/s
                                  </span>
                                </>
                              )}
                            </div>
                          </div>
                        </div>

                        <div
                          className="flex items-center gap-1.5"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <Badge
                            variant="outline"
                            className="text-[10px] font-mono mr-1 gap-1.5 font-normal"
                          >
                            <span
                              className={`h-2 w-2 rounded-full ${
                                isOffline ? "bg-destructive" : "bg-emerald-500 animate-pulse"
                              }`}
                            />
                            {formatHeartbeatAge(w.lastHeartbeatAt)}
                          </Badge>

                          {!isOffline ? (
                            <>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() => onTriggerWatchdog(w.id, w.gpuName)}
                                    className="h-7 text-[11px] px-2"
                                  >
                                    <Wrench className="w-3 h-3 mr-1" />
                                    Verifica
                                  </Button>
                                </TooltipTrigger>
                                <TooltipContent>
                                  Verifica riavvio automatico e connettività del nodo GPU
                                </TooltipContent>
                              </Tooltip>

                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() => onDisconnectWorker(w.id, w.nodeName)}
                                    className="h-7 text-[11px] px-2 text-destructive hover:text-destructive"
                                  >
                                    <Unplug className="w-3 h-3 mr-1" />
                                    Scollega
                                  </Button>
                                </TooltipTrigger>
                                <TooltipContent>
                                  Scollega immediatamente questa GPU
                                </TooltipContent>
                              </Tooltip>
                            </>
                          ) : (
                            <Button
                              variant="destructive"
                              size="sm"
                              onClick={() => onDeleteWorker(w.id, w.nodeName)}
                              className="h-7 text-[11px] px-2.5"
                            >
                              <Trash2 className="w-3 h-3 mr-1" />
                              Rimuovi Nodo Offline
                            </Button>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
              {workerTotalPages > 1 && (
                <div className="flex items-center justify-between pt-2 text-xs text-muted-foreground">
                  <span className="font-mono">
                    Pagina {safeWorkerPage} di {workerTotalPages} ({filteredWorkers.length} nodi)
                  </span>
                  <div className="flex items-center gap-1.5">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={safeWorkerPage <= 1}
                      onClick={() => setWorkerPage((p) => Math.max(1, p - 1))}
                      className="h-7 text-xs px-2"
                    >
                      <ChevronLeft className="w-3.5 h-3.5" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={safeWorkerPage >= workerTotalPages}
                      onClick={() => setWorkerPage((p) => Math.min(workerTotalPages, p + 1))}
                      className="h-7 text-xs px-2"
                    >
                      <ChevronRight className="w-3.5 h-3.5" />
                    </Button>
                  </div>
                </div>
              )}
              </>
            )}
          </CardContent>
        </Card>

        {/* Right: Cluster Composer & Preview */}
        <Card className="lg:col-span-5">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">
              {isEn ? "Cluster Composer & Resource Preview" : "Configuratore Cluster & Anteprima Risorse"}
            </CardTitle>
            <CardDescription>
              {isEn
                ? "Preview layer partitioning in advance and launch the cluster in 1 click."
                : "Calcola in anticipo la partizione dei layer e avvia il cluster con 1 click."}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label className="text-xs font-medium">
                  {isEn ? "Model to Serve" : "Modello da Servire"}
                </Label>
                <Select value={selectedModelId} onValueChange={setSelectedModelId}>
                  <SelectTrigger className="h-9 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {models.map((m) => (
                      <SelectItem key={m.modelId} value={m.modelId}>
                        {m.displayName} ({m.requiredVramMb} MB)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs font-medium">
                  {isEn ? "Cluster Name (optional)" : "Nome Cluster (opzionale)"}
                </Label>
                <Input
                  value={customRigName}
                  onChange={(e) => setCustomRigName(e.target.value)}
                  placeholder={isEn ? "Cluster name" : "Nome cluster"}
                  className="h-9 text-xs"
                />
              </div>
            </div>

            {!topologyPreview ? (
              <Alert className="border-dashed bg-muted/20">
                <Activity className="h-4 w-4" />
                <AlertDescription className="text-xs text-muted-foreground">
                  {isEn
                    ? "Select at least one online GPU on the left to compute topology."
                    : "Seleziona almeno una GPU online nella colonna a sinistra per calcolare la topologia."}
                </AlertDescription>
              </Alert>
            ) : (
              <Card className="bg-muted/20 shadow-none">
                <CardContent className="p-3.5 space-y-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">
                      {isEn ? "Combined Allocatable VRAM:" : "VRAM Allocabile Combinata:"}
                    </span>
                    <span className="font-mono font-semibold">
                      {(topologyPreview.totalAllocatableMb / 1024).toFixed(2)} GB (
                      {isEn ? "Required" : "Richiesti"}:{" "}
                      {(topologyPreview.requiredMb / 1024).toFixed(2)} GB)
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">
                      {isEn ? "`--tensor-split` Parameter:" : "Parametro `--tensor-split`:"}
                    </span>
                    <Badge variant="secondary" className="font-mono text-xs">
                      {topologyPreview.tensorSplitCsv}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">
                      {isEn ? "Unlocked Context Window (`-c`):" : "Finestra Contesto Sbloccata (`-c`):"}
                    </span>
                    <Badge variant="outline" className="font-mono text-xs text-emerald-500 border-emerald-500/40">
                      {topologyPreview.unlockedContext.toLocaleString()} {isEn ? "tokens" : "token"}
                    </Badge>
                  </div>

                  <Separator />

                  <div className="space-y-2">
                    <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                      {isEn
                        ? `Layer Partition per Node (${topologyPreview.stages.length} Stages)`
                        : `Partizione Layer per Singolo Nodo (${topologyPreview.stages.length} Stadi)`}
                    </div>
                    <Card className="shadow-none overflow-hidden">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead className="h-7 text-[10px]">
                              {isEn ? "Node / Stage" : "Nodo / Stage"}
                            </TableHead>
                            <TableHead className="h-7 text-[10px]">
                              {isEn ? "Allocated VRAM" : "VRAM Allocata"}
                            </TableHead>
                            <TableHead className="h-7 text-[10px] text-right">Layer</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {topologyPreview.stages.map((st) => (
                            <TableRow key={st.worker.id}>
                              <TableCell className="py-1.5 text-xs">
                                <div className="font-semibold">
                                  #{st.stageIndex} {st.worker.nodeName}
                                </div>
                                <div className="text-[10px] text-muted-foreground">
                                  {st.worker.gpuName}
                                </div>
                              </TableCell>
                              <TableCell className="py-1.5 font-mono text-[11px] text-muted-foreground">
                                {st.splitGb} GB
                              </TableCell>
                              <TableCell className="py-1.5 text-right">
                                <Badge variant="outline" className="font-mono text-[10px]">
                                  {st.layerStart}→{st.layerEnd} ({st.fractionPct}%)
                                </Badge>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </Card>
                  </div>

                  <Button
                    onClick={handleApplyTopology}
                    disabled={!topologyPreview.enoughVram || reconfiguring}
                    className="w-full h-9 text-xs"
                  >
                    <RefreshCw
                      className={`w-3.5 h-3.5 mr-1.5 ${reconfiguring ? "animate-spin" : ""}`}
                    />
                    {reconfiguring
                      ? isEn
                        ? "Activating Cluster..."
                        : "Attivazione Cluster in corso..."
                      : isEn
                      ? `Start Cluster (${topologyPreview.stages.length} GPU)`
                      : `Avvia Cluster (${topologyPreview.stages.length} GPU)`}
                  </Button>
                </CardContent>
              </Card>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
