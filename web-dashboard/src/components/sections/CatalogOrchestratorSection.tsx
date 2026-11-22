"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  CatalogModel,
  GpuWorkerNode,
  ManagedRig
} from "@/types/control-plane";
import { useI18n } from "@/i18n/translations";
import {
  createManualRig,
  deleteCatalogModel,
  dissolveManagedRig,
  registerCatalogModel,
  toggleModelOrchestrationMode,
  updateModelPriorityOrder
} from "@/services/control-plane-api";
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
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
import {
  Check,
  Code2,
  Copy,
  Download,
  Heart,
  Loader2,
  MessageSquare,
  Plus,
  Search,
  Trash2,
  Unlock
} from "lucide-react";

interface CatalogOrchestratorSectionProps {
  t: Record<string, string>;
  models: CatalogModel[];
  standbyWorkers: GpuWorkerNode[];
  activeRigs: ManagedRig[];
  isAdmin?: boolean;
  onSelectModelForPlayground?: (modelId: string) => void;
  onNavigateToApiKeys?: () => void;
  onNotify: (msg: string) => void;
  onRefresh: () => void;
}

interface HfRepoSummary {
  id: string;
  downloads?: number;
  likes?: number;
}

function parseHuggingFaceUrlMetadata(url: string): {
  modelId: string;
  displayName: string;
  localCachePath: string;
} | null {
  const trimmed = url.trim();
  if (!trimmed) return null;
  const parts = trimmed.split("/").filter(Boolean);
  const lastSegment = parts[parts.length - 1] || "";
  const filename = lastSegment.split("?")[0];
  if (!filename) return null;

  const baseName = filename.replace(/\.gguf$/i, "");
  const cleanTitle = baseName
    .replace(/[-_]+/g, " ")
    .replace(/\b(Q[0-9]_[A-Z0-9_]+|IQ[0-9]_[A-Z0-9_]+|F16|BF16)\b/gi, "")
    .replace(/\s+/g, " ")
    .trim();

  const slug = cleanTitle
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return {
    modelId: slug || "custom-model",
    displayName: cleanTitle || baseName,
    localCachePath: `models/${filename.endsWith(".gguf") ? filename : `${filename}.gguf`}`
  };
}

function getProviderLabel(displayName: string, modelId: string): string {
  const s = `${displayName} ${modelId}`.toLowerCase();
  if (s.includes("nemotron") || s.includes("nvidia")) return "NVIDIA";
  if (s.includes("llama") || s.includes("meta")) return "Meta";
  if (s.includes("qwen")) return "Qwen";
  if (s.includes("deepseek")) return "DeepSeek";
  if (s.includes("mistral") || s.includes("mixtral")) return "Mistral AI";
  if (s.includes("gemma") || s.includes("google")) return "Google";
  if (s.includes("phi")) return "Microsoft";
  return "Open Weights";
}

function cleanDisplayTitle(name: string): string {
  return name.replace(/\s*\([Qq][0-9][^)]*\)/g, "").trim();
}

function formatCompactCount(n?: number): string {
  if (!n) return "0";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function pickPreferredGgufFile(files: string[]): string {
  if (files.length === 0) return "";
  const priorities = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q4_0", "Q8_0"];
  for (const p of priorities) {
    const match = files.find((f) => f.toUpperCase().includes(p));
    if (match) return match;
  }
  return files[0];
}

export function CatalogOrchestratorSection({
  models,
  standbyWorkers,
  activeRigs,
  isAdmin = false,
  onSelectModelForPlayground,
  onNavigateToApiKeys,
  onNotify,
  onRefresh
}: CatalogOrchestratorSectionProps) {
  const { locale } = useI18n();
  const isEn = locale === "en";
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [statusFilter, setStatusFilter] = useState<"ALL" | "ONLINE">("ALL");
  const [copiedModelId, setCopiedModelId] = useState<string | null>(null);

  // Add Model Modal + Live Hugging Face Hub Search
  const [addModelModalOpen, setAddModelModalOpen] = useState<boolean>(false);
  const [hfSearchQuery, setHfSearchQuery] = useState<string>("");
  const [hfResults, setHfResults] = useState<HfRepoSummary[]>([]);
  const [hfSearching, setHfSearching] = useState<boolean>(false);
  const [selectedHfRepo, setSelectedHfRepo] = useState<string>("");
  const [repoGgufFiles, setRepoGgufFiles] = useState<string[]>([]);
  const [selectedGgufFile, setSelectedGgufFile] = useState<string>("");
  const [loadingRepoFiles, setLoadingRepoFiles] = useState<boolean>(false);

  const [newModelHfUrl, setNewModelHfUrl] = useState<string>("");
  const [newModelName, setNewModelName] = useState<string>("");
  const [newModelId, setNewModelId] = useState<string>("");
  const [autoPilotEnabled, setAutoPilotEnabled] = useState<boolean>(true);
  const [submittingModel, setSubmittingModel] = useState<boolean>(false);

  // Optional Admin Rig Assignment
  const [selectedModelForRig, setSelectedModelForRig] = useState<string>(
    models[0]?.modelId || "qwen3.8-27b"
  );
  const [selectedWorkerIds, setSelectedWorkerIds] = useState<string[]>([]);

  // Live query to Hugging Face Hub API when modal is open
  useEffect(() => {
    if (!addModelModalOpen) return;
    let cancelled = false;
    const timer = setTimeout(async () => {
      setHfSearching(true);
      try {
        const q = hfSearchQuery.trim();
        const url = q
          ? `https://huggingface.co/api/models?search=${encodeURIComponent(
              q
            )}&filter=gguf&sort=downloads&direction=-1&limit=10`
          : `https://huggingface.co/api/models?filter=gguf&sort=downloads&direction=-1&limit=10`;
        const res = await fetch(url);
        if (res.ok && !cancelled) {
          const data: HfRepoSummary[] = await res.json();
          setHfResults(data);
        }
      } catch {
        // Ignore network errors
      } finally {
        if (!cancelled) setHfSearching(false);
      }
    }, 280);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [addModelModalOpen, hfSearchQuery]);

  const handleSelectHfRepo = async (repoId: string) => {
    setSelectedHfRepo(repoId);
    setLoadingRepoFiles(true);
    setRepoGgufFiles([]);
    try {
      const res = await fetch(`https://huggingface.co/api/models/${repoId}`);
      if (res.ok) {
        const data = await res.json();
        const siblings: { rfilename: string }[] = data.siblings || [];
        const ggufs = siblings
          .map((s) => s.rfilename)
          .filter(
            (f) =>
              f.toLowerCase().endsWith(".gguf") &&
              !f.toLowerCase().includes("mmproj") &&
              !f.toLowerCase().includes("-00002-of-") &&
              !f.toLowerCase().includes("-00003-of-")
          );

        setRepoGgufFiles(ggufs);
        const preferred = pickPreferredGgufFile(ggufs);
        if (preferred) {
          applySelectedRepoAndFile(repoId, preferred);
        } else {
          const fallbackUrl = `https://huggingface.co/${repoId}`;
          setNewModelHfUrl(fallbackUrl);
          const repoShort = repoId.split("/")[1] || repoId;
          const clean = repoShort.replace(/-GGUF$/i, "").replace(/[-_]+/g, " ");
          setNewModelName(clean);
          setNewModelId(clean.toLowerCase().replace(/[^a-z0-9]+/g, "-"));
        }
      }
    } finally {
      setLoadingRepoFiles(false);
    }
  };

  const applySelectedRepoAndFile = (repoId: string, filename: string) => {
    setSelectedGgufFile(filename);
    const fullUrl = `https://huggingface.co/${repoId}/resolve/main/${filename}`;
    setNewModelHfUrl(fullUrl);
    const parsed = parseHuggingFaceUrlMetadata(fullUrl);
    if (parsed) {
      setNewModelName(parsed.displayName);
      setNewModelId(parsed.modelId);
    }
  };

  const handleHfUrlChange = (url: string) => {
    setNewModelHfUrl(url);
    const parsed = parseHuggingFaceUrlMetadata(url);
    if (parsed) {
      setNewModelName(parsed.displayName);
      setNewModelId(parsed.modelId);
    }
  };

  const handleRegisterNewModel = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newModelName.trim() || !newModelId.trim() || !newModelHfUrl.trim()) return;
    setSubmittingModel(true);
    const parsed = parseHuggingFaceUrlMetadata(newModelHfUrl);
    const localPath = parsed?.localCachePath || `models/${newModelId.trim()}.gguf`;

    const ok = await registerCatalogModel({
      modelId: newModelId.trim(),
      displayName: newModelName.trim(),
      huggingfaceRepo: newModelHfUrl.trim(),
      baseGgufPath: localPath,
      securityProfile: "FULL_COVARIANT_SHIELD",
      orchestrationMode: autoPilotEnabled ? "AUTO_PILOT" : "MANUAL_ONLY"
    });
    setSubmittingModel(false);

    if (ok) {
      setAddModelModalOpen(false);
      setNewModelHfUrl("");
      setNewModelName("");
      setNewModelId("");
      setSelectedHfRepo("");
      setRepoGgufFiles([]);
      onNotify(`Modello '${newModelName.trim()}' aggiunto al catalogo`);
      onRefresh();
    } else {
      onNotify("Impossibile registrare il modello. Verifica che l'ID non sia duplicato.");
    }
  };

  const handleDeleteModel = async (m: CatalogModel) => {
    const ok = await deleteCatalogModel(m.modelId);
    if (ok) {
      onNotify(`Modello '${m.displayName}' rimosso dal catalogo`);
      onRefresh();
    }
  };

  const handleCopyModelId = (modelId: string) => {
    navigator.clipboard.writeText(modelId);
    setCopiedModelId(modelId);
    setTimeout(() => setCopiedModelId(null), 2000);
  };

  const handleToggleOrchestration = async (model: CatalogModel) => {
    await toggleModelOrchestrationMode(model);
    onNotify(`Modalità aggiornata per ${cleanDisplayTitle(model.displayName)}`);
    onRefresh();
  };

  const handlePriorityChange = async (model: CatalogModel, nextPriority: number) => {
    await updateModelPriorityOrder(model, nextPriority);
    onNotify(
      `Priorità P${nextPriority} applicata a ${cleanDisplayTitle(model.displayName)}`
    );
    onRefresh();
  };

  const handleCreateManual = async () => {
    if (selectedWorkerIds.length === 0) return;
    const rigName = `rig-${selectedModelForRig}-${selectedWorkerIds.length}gpu`;
    const res = await createManualRig(rigName, selectedModelForRig, selectedWorkerIds);
    if (res.ok) {
      setSelectedWorkerIds([]);
      onNotify(`Modello attivato su ${selectedWorkerIds.length} GPU`);
      onRefresh();
    } else {
      onNotify(`Errore: ${res.errorMessage}`);
    }
  };

  const handleDissolve = async (rig: ManagedRig) => {
    await dissolveManagedRig(rig.id);
    onNotify(`GPU rilasciate da '${rig.rigName}'`);
    onRefresh();
  };

  const filteredModels = useMemo(() => {
    return models.filter((m) => {
      const q = searchQuery.trim().toLowerCase();
      const matchesSearch =
        !q ||
        m.displayName.toLowerCase().includes(q) ||
        m.modelId.toLowerCase().includes(q) ||
        getProviderLabel(m.displayName, m.modelId).toLowerCase().includes(q);
      if (!matchesSearch) return false;

      if (statusFilter === "ONLINE") {
        return activeRigs.some(
          (r) => r.modelId === m.modelId && r.lifecycleState === "SERVING"
        );
      }
      return true;
    });
  }, [models, searchQuery, statusFilter, activeRigs]);

  return (
    <div className="space-y-6">
      {/* Header OpenRouter-Style */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold">
            {isEn ? "Available AI Models" : "Modelli AI Disponibili"}
          </h2>
          <p className="text-xs text-muted-foreground">
            {isEn
              ? "Browse the model catalog, copy the Model ID for OpenAI API calls, or test them live in chat."
              : "Esplora il catalogo dei modelli, copia l'identificativo per le tue chiamate API OpenAI o provali subito in chat."}
          </p>
        </div>

        {isAdmin && (
          <Dialog open={addModelModalOpen} onOpenChange={setAddModelModalOpen}>
            <DialogTrigger asChild>
              <Button className="gap-2 h-9 text-xs">
                <Plus className="h-4 w-4" />
                {isEn ? "Add Model to Catalog" : "Aggiungi Modello al Catalogo"}
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-xl">
              <DialogHeader>
                <DialogTitle>
                  {isEn ? "Search & Select Model" : "Cerca e Seleziona Modello"}
                </DialogTitle>
                <DialogDescription className="text-xs">
                  {isEn
                    ? "Search public model repositories and click a model to auto-fill link, quantization, and API ID."
                    : "Cerca nel repository modelli pubblici e clicca sul modello per compilare automaticamente link, variante e ID API."}
                </DialogDescription>
              </DialogHeader>

              <form onSubmit={handleRegisterNewModel} className="space-y-4 py-1">
                {/* LIVE MODEL HUB SEARCH */}
                <div className="space-y-2">
                  <Label className="text-xs">
                    {isEn ? "Search Model Repository" : "Cerca nel Repository Modelli"}
                  </Label>
                  <div className="relative">
                    <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-muted-foreground" />
                    <Input
                      value={hfSearchQuery}
                      onChange={(e) => setHfSearchQuery(e.target.value)}
                      placeholder={isEn ? "Search GGUF model..." : "Cerca modello GGUF..."}
                      className="pl-8 h-9 text-xs"
                    />
                    {hfSearching && (
                      <Loader2 className="w-3.5 h-3.5 animate-spin absolute right-3 top-2.5 text-muted-foreground" />
                    )}
                  </div>

                  {/* Live Results List from Hugging Face */}
                  <ScrollArea className="h-44 rounded-lg border bg-muted/20">
                    <div className="divide-y">
                      {hfResults.length === 0 && !hfSearching ? (
                        <div className="p-3 text-center text-xs text-muted-foreground">
                          {isEn ? "No models found for this search." : "Nessun modello trovato per questa ricerca."}
                        </div>
                      ) : (
                        hfResults.map((repo) => {
                          const isSelected = selectedHfRepo === repo.id;
                          return (
                            <Button
                              key={repo.id}
                              type="button"
                              variant={isSelected ? "secondary" : "ghost"}
                              onClick={() => handleSelectHfRepo(repo.id)}
                              className="w-full h-9 rounded-none px-3 flex items-center justify-between gap-2 text-xs font-normal"
                            >
                              <span className="truncate font-mono">{repo.id}</span>
                              <span className="flex items-center gap-2 text-[11px] text-muted-foreground shrink-0">
                                <Badge variant="outline" className="h-5 px-1.5 gap-1 font-mono text-[10px]">
                                  <Download className="w-2.5 h-2.5" />
                                  {formatCompactCount(repo.downloads)}
                                </Badge>
                                <Badge variant="outline" className="h-5 px-1.5 gap-1 font-mono text-[10px]">
                                  <Heart className="w-2.5 h-2.5" />
                                  {formatCompactCount(repo.likes)}
                                </Badge>
                              </span>
                            </Button>
                          );
                        })
                      )}
                    </div>
                  </ScrollArea>
                </div>

                {/* Quantization File Selector (if repo has multiple .gguf files) */}
                {loadingRepoFiles && (
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Lettura file disponibili nel repository {selectedHfRepo}...
                  </div>
                )}

                {repoGgufFiles.length > 0 && !loadingRepoFiles && (
                  <div className="space-y-1.5">
                    <Label className="text-xs">
                      Variante / Quantizzazione disponibile nel repository ({repoGgufFiles.length} file)
                    </Label>
                    <Select
                      value={selectedGgufFile}
                      onValueChange={(file) => applySelectedRepoAndFile(selectedHfRepo, file)}
                    >
                      <SelectTrigger className="h-9 text-xs font-mono">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {repoGgufFiles.map((f) => (
                          <SelectItem key={f} value={f} className="text-xs font-mono">
                            {f}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <Label className="text-xs">
                      {isEn ? "Catalog Display Name" : "Nome Visualizzato nel Catalogo"}
                    </Label>
                    <Input
                      value={newModelName}
                      onChange={(e) => setNewModelName(e.target.value)}
                      placeholder={isEn ? "Display name" : "Nome visualizzato"}
                      className="h-9 text-xs"
                      required
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label className="text-xs">
                      {isEn ? "Model ID (for /v1 API calls)" : "ID Modello (per chiamate API /v1)"}
                    </Label>
                    <Input
                      value={newModelId}
                      onChange={(e) => setNewModelId(e.target.value)}
                      placeholder={isEn ? "Unique model ID" : "ID univoco modello"}
                      className="h-9 text-xs font-mono"
                      required
                    />
                  </div>
                </div>

                <div className="space-y-1.5">
                  <Label className="text-xs text-muted-foreground">
                    {isEn
                      ? "Model Repository URL (auto-filled or manual)"
                      : "URL Repository Modello (compilato automaticamente o manuale)"}
                  </Label>
                  <Input
                    value={newModelHfUrl}
                    onChange={(e) => handleHfUrlChange(e.target.value)}
                    placeholder="https://.../resolve/main/model.gguf"
                    className="h-8 text-[11px] font-mono"
                    required
                  />
                </div>

                <div className="flex items-center justify-between rounded-lg border p-3 bg-muted/30">
                  <div className="space-y-0.5">
                    <div className="text-xs font-medium">
                      {isEn ? "Auto-Start on GPUs" : "Avvio Automatico sulle GPU"}
                    </div>
                    <div className="text-[11px] text-muted-foreground">
                      {isEn
                        ? "Automatically schedules the model whenever free GPUs are available."
                        : "Assegna automaticamente il modello quando ci sono GPU libere."}
                    </div>
                  </div>
                  <Switch
                    checked={autoPilotEnabled}
                    onCheckedChange={setAutoPilotEnabled}
                  />
                </div>

                <DialogFooter>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => setAddModelModalOpen(false)}
                  >
                    {isEn ? "Cancel" : "Annulla"}
                  </Button>
                  <Button
                    type="submit"
                    size="sm"
                    disabled={submittingModel || !newModelHfUrl.trim()}
                  >
                    {submittingModel
                      ? isEn
                        ? "Adding..."
                        : "Aggiunta in corso..."
                      : isEn
                      ? "Add to Catalog"
                      : "Aggiungi al Catalogo"}
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
        )}
      </div>

      {/* Search & Filter Bar (OpenRouter Style) */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 absolute left-3 top-2.5 text-muted-foreground" />
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={isEn ? "Search model..." : "Cerca modello..."}
            className="pl-9 h-9 text-xs"
          />
        </div>
        <div className="flex items-center gap-1.5">
          <Button
            variant={statusFilter === "ALL" ? "default" : "outline"}
            size="sm"
            onClick={() => setStatusFilter("ALL")}
            className="h-8 text-xs"
          >
            {isEn ? `All Models (${models.length})` : `Tutti i Modelli (${models.length})`}
          </Button>
          <Button
            variant={statusFilter === "ONLINE" ? "default" : "outline"}
            size="sm"
            onClick={() => setStatusFilter("ONLINE")}
            className="h-8 text-xs gap-1.5"
          >
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            {isEn ? "Active Now" : "Attivi Ora"}
          </Button>
        </div>
      </div>

      {/* Clean OpenRouter-Style Model Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {filteredModels.map((m) => {
          const servingRig = activeRigs.find(
            (r) => r.modelId === m.modelId && r.lifecycleState === "SERVING"
          );
          const isOnline = !!servingRig;
          const provider = getProviderLabel(m.displayName, m.modelId);
          const contextTokens = servingRig?.autoMaximizedContextSize || 32768;
          const contextLabel = `${Math.round(contextTokens / 1024)}K context`;

          return (
            <Card key={m.modelId} className="flex flex-col justify-between">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-2">
                  <div className="space-y-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant="secondary" className="text-[10px] font-medium">
                        {provider}
                      </Badge>
                      <Badge variant="outline" className="text-[10px] font-mono">
                        {(m.requiredVramMb / 1024).toFixed(1)} GB VRAM
                      </Badge>
                      <Badge variant="outline" className="text-[10px] font-mono">
                        {contextLabel}
                      </Badge>
                    </div>
                    <CardTitle className="text-base pt-0.5">
                      {cleanDisplayTitle(m.displayName)}
                    </CardTitle>
                  </div>

                  {servingRig ? (
                    servingRig.inferenceEngineReady ? (
                      <Badge className="bg-emerald-600 hover:bg-emerald-600 text-white gap-1.5 text-[10px] shrink-0 font-mono">
                        <span className="h-1.5 w-1.5 rounded-full bg-white animate-pulse" />
                        {isEn ? "4/4 · Operational (200 OK)" : "4/4 · Operativo (200 OK)"}
                      </Badge>
                    ) : servingRig.runtimePhase === "LOADING_VRAM" ? (
                      <Badge className="bg-blue-600 hover:bg-blue-600 text-white gap-1.5 text-[10px] shrink-0 font-mono">
                        <Loader2 className="h-3 w-3 animate-spin" />
                        {isEn ? "3/4 · Loading Model..." : "3/4 · Caricamento Modello..."}
                      </Badge>
                    ) : servingRig.runtimePhase === "PREWARMING_SHARDS" ? (
                      <Badge className="bg-indigo-600 hover:bg-indigo-600 text-white gap-1.5 text-[10px] shrink-0 font-mono">
                        <Loader2 className="h-3 w-3 animate-spin" />
                        {isEn ? "2/4 · Preparing Weights..." : "2/4 · Preparazione Pesi..."}
                      </Badge>
                    ) : (
                      <Badge
                        variant="outline"
                        className="border-amber-500/60 bg-amber-500/10 text-amber-400 gap-1.5 text-[10px] shrink-0 font-mono"
                      >
                        <span className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse" />
                        {isEn ? "1/4 · Layers Allocated" : "1/4 · Layer Allocati"}
                      </Badge>
                    )
                  ) : (
                    <Badge variant="secondary" className="gap-1.5 text-[10px] shrink-0">
                      <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground" />
                      Standby
                    </Badge>
                  )}
                </div>
              </CardHeader>

              <CardContent className="space-y-3 pb-3">
                {/* Copyable Model ID Bar (like OpenRouter) */}
                <div className="flex items-center justify-between rounded-md border bg-muted/40 px-3 py-1.5">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold shrink-0">
                      Model ID
                    </span>
                    <code className="text-xs font-mono truncate">{m.modelId}</code>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => handleCopyModelId(m.modelId)}
                    className="h-6 px-2 text-[11px] gap-1 shrink-0"
                  >
                    {copiedModelId === m.modelId ? (
                      <>
                        <Check className="h-3 w-3 text-emerald-500" />
                        {isEn ? "Copied" : "Copiato"}
                      </>
                    ) : (
                      <>
                        <Copy className="h-3 w-3" />
                        {isEn ? "Copy" : "Copia"}
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>

              <CardFooter className="flex flex-wrap items-center justify-between gap-2 border-t pt-3">
                <div className="flex items-center gap-2">
                  {onSelectModelForPlayground && (
                    <Button
                      size="sm"
                      onClick={() => onSelectModelForPlayground(m.modelId)}
                      className="h-8 text-xs gap-1.5"
                    >
                      <MessageSquare className="h-3.5 w-3.5" />
                      {isEn ? "Try in Chat" : "Prova in Chat"}
                    </Button>
                  )}
                  {onNavigateToApiKeys && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        handleCopyModelId(m.modelId);
                        onNavigateToApiKeys();
                      }}
                      className="h-8 text-xs gap-1.5"
                    >
                      <Code2 className="h-3.5 w-3.5" />
                      {isEn ? "Use via API" : "Usa via API"}
                    </Button>
                  )}
                </div>

                {isAdmin && (
                  <div className="flex items-center gap-2">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div>
                          <Select
                            value={String(Math.min(5, Math.max(1, m.priorityOrder || 1)))}
                            onValueChange={(val) => handlePriorityChange(m, Number(val))}
                          >
                            <SelectTrigger className="h-7 w-[92px] text-[11px] font-mono px-2">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="1" className="text-xs font-mono">
                                P1 (Max)
                              </SelectItem>
                              <SelectItem value="2" className="text-xs font-mono">
                                P2
                              </SelectItem>
                              <SelectItem value="3" className="text-xs font-mono">
                                P3
                              </SelectItem>
                              <SelectItem value="4" className="text-xs font-mono">
                                P4
                              </SelectItem>
                              <SelectItem value="5" className="text-xs font-mono">
                                P5
                              </SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        Priorità di allocazione GPU (P1 = Massima priorità con promozione automatica)
                      </TooltipContent>
                    </Tooltip>

                    <div className="flex items-center gap-1.5">
                      <Switch
                        checked={m.orchestrationMode === "AUTO_PILOT"}
                        onCheckedChange={() => handleToggleOrchestration(m)}
                      />
                      <Label className="text-[11px] text-muted-foreground font-normal">
                        Auto
                      </Label>
                    </div>
                    {models.length > 1 && (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            onClick={() => handleDeleteModel(m)}
                            className="h-7 w-7 text-muted-foreground hover:text-destructive"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>Rimuovi modello dal catalogo</TooltipContent>
                      </Tooltip>
                    )}
                  </div>
                )}
              </CardFooter>
            </Card>
          );
        })}
      </div>

      {/* Admin-Only Simple GPU Rig Assignment */}
      {isAdmin && (standbyWorkers.length > 0 || activeRigs.length > 0) && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">
              {isEn ? "Quick Active Instance Management (Admin)" : "Gestione Rapida Istanze Attive (Admin)"}
            </CardTitle>
            <CardDescription className="text-xs">
              {isEn
                ? "Launch a model on available GPUs or release active instances."
                : "Avvia un modello sulle GPU disponibili oppure rilascia le istanze attive."}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {standbyWorkers.length > 0 && (
              <div className="flex flex-wrap items-center gap-2">
                {standbyWorkers.map((sw) => {
                  const isSel = selectedWorkerIds.includes(sw.id);
                  return (
                    <Button
                      key={sw.id}
                      type="button"
                      size="sm"
                      variant={isSel ? "default" : "outline"}
                      onClick={() =>
                        setSelectedWorkerIds((prev) =>
                          isSel ? prev.filter((i) => i !== sw.id) : [...prev, sw.id]
                        )
                      }
                      className="text-xs h-8"
                    >
                      {sw.gpuName} ({sw.nodeName})
                    </Button>
                  );
                })}
                <Select
                  value={selectedModelForRig}
                  onValueChange={setSelectedModelForRig}
                >
                  <SelectTrigger className="h-8 w-[240px] text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {models.map((m) => (
                      <SelectItem key={m.modelId} value={m.modelId} className="text-xs">
                        {cleanDisplayTitle(m.displayName)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button
                  size="sm"
                  disabled={selectedWorkerIds.length === 0}
                  onClick={handleCreateManual}
                  className="h-8 text-xs"
                >
                  {isEn
                    ? `Start Model (${selectedWorkerIds.length} GPU)`
                    : `Avvia Modello (${selectedWorkerIds.length} GPU)`}
                </Button>
              </div>
            )}

            {activeRigs.length > 0 && (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{isEn ? "Active Model" : "Modello Attivo"}</TableHead>
                    <TableHead>{isEn ? "Allocated GPUs" : "GPU Allocate"}</TableHead>
                    <TableHead>{isEn ? "Requests Served" : "Richieste Servite"}</TableHead>
                    <TableHead className="text-right">{isEn ? "Actions" : "Azioni"}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {activeRigs.map((rig) => (
                    <TableRow key={rig.id}>
                      <TableCell className="font-semibold text-xs">
                        <span className="inline-flex items-center gap-2">
                          <span className="h-2 w-2 rounded-full bg-emerald-500" />
                          {rig.modelDisplayName}
                        </span>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className="font-mono text-[10px]">
                          {rig.allocations.length} GPU
                        </Badge>
                      </TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {rig.servedRequestsCount} {isEn ? "requests" : "richieste"}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleDissolve(rig)}
                          className="h-7 text-xs gap-1 text-destructive hover:text-destructive"
                        >
                          <Unlock className="h-3 w-3" />
                          {isEn ? "Stop & Release GPUs" : "Ferma e Rilascia GPU"}
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
