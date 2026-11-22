"use client";

import React, { useState } from "react";
import {
  EnrollmentTokenIssued,
  EnrollmentTokenItem,
  GpuWorkerNode
} from "@/types/control-plane";
import { useI18n } from "@/i18n/translations";
import {
  API_BASE,
  issueEnrollmentToken,
  revokeEnrollmentToken,
  triggerWatchdogFault
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
import { CodeBlock } from "@/components/ui/chat-markdown";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CheckCircle2, Copy, Cpu, Plus, ShieldCheck, Terminal, Trash2 } from "lucide-react";

interface GpuProviderSectionProps {
  t: Record<string, string>;
  workers: GpuWorkerNode[];
  enrollmentTokens: EnrollmentTokenItem[];
  copiedId: string | null;
  onCopyText: (text: string, id: string) => void;
  onNotify: (msg: string) => void;
  onRefresh: () => void;
}

export function GpuProviderSection({
  t,
  workers,
  enrollmentTokens,
  copiedId,
  onCopyText,
  onNotify,
  onRefresh
}: GpuProviderSectionProps) {
  const [gpuModalOpen, setGpuModalOpen] = useState<boolean>(false);
  const [tokenLabel, setTokenLabel] = useState<string>("");
  const [poolMode, setPoolMode] = useState<"SHARED_POOL" | "PRIVATE_SOLO">("SHARED_POOL");
  const [powerPct, setPowerPct] = useState<number>(100);
  const [issuedToken, setIssuedToken] = useState<EnrollmentTokenIssued | null>(null);
  const [hasCopiedIssuedToken, setHasCopiedIssuedToken] = useState<boolean>(false);
  const [showRevokedTokens, setShowRevokedTokens] = useState<boolean>(false);
  const [savedCmdByTokenId, setSavedCmdByTokenId] = useState<Record<string, string>>(() => {
    if (typeof window === "undefined") return {};
    try {
      // Purge any legacy persisted commands from localStorage; keep only session-scoped cache
      window.localStorage.removeItem("cutmesh_issued_token_cmds_v1");
      const raw = window.sessionStorage.getItem("cutmesh_issued_token_cmds_v1");
      return raw ? JSON.parse(raw) : {};
    } catch {
      return {};
    }
  });

  const revokedTokensCount = enrollmentTokens.filter((tk) => tk.revoked).length;
  const visibleTokens = showRevokedTokens
    ? enrollmentTokens
    : enrollmentTokens.filter((tk) => !tk.revoked);

  const buildFallbackDockerCmd = (tk: EnrollmentTokenItem) => {
    const safeName = tk.label.replace(/[^a-zA-Z0-9_-]/g, "-") || "cutmesh-gpu-node";
    return `docker run -d --gpus all --restart unless-stopped --name ${safeName} -p 40000:40000/udp -v cutmesh-cache:/data/cutmesh -e CUTMESH_QUIC_PORT=40000 -e CUTMESH_HUB_URL="${API_BASE}" -e CUTMESH_NODE_TOKEN="<PASTE_TOKEN_${tk.tokenPrefix}>" -e CUTMESH_NODE_NAME="${safeName}" -e CUTMESH_POOL_MODE="${tk.defaultPoolMode}" -e CUTMESH_POWER_PCT=100 -e CUTMESH_STRICT_DIRECT=1 ghcr.io/massimolauri/cutmesh-worker:latest`;
  };

  // Dynamically resolve any localhost/host.docker.internal hub URL to the real reachable Control Plane URL
  // and inject the selected power budget into Docker commands.
  const normalizeHubCommand = (cmd: string, isDocker: boolean): string => {
    let out = cmd.replace(
      /https?:\/\/(localhost|127\.0\.0\.1|host\.docker\.internal)(:\d+)?/gi,
      API_BASE
    );
    if (isDocker) {
      if (out.includes("CUTMESH_POWER_PCT=")) {
        out = out.replace(/CUTMESH_POWER_PCT=\d+/g, `CUTMESH_POWER_PCT=${powerPct}`);
      } else if (out.includes("SYNAPSE_POWER_PCT=")) {
        out = out.replace(/SYNAPSE_POWER_PCT=\d+/g, `SYNAPSE_POWER_PCT=${powerPct}`);
      } else {
        out = out.replace(
          /-e (CUTMESH|SYNAPSE)_POOL_MODE=/,
          `-e $1_POWER_PCT=${powerPct} -e $1_POOL_MODE=`
        );
      }
    }
    return out;
  };

  const handleIssueToken = async () => {
    const label = tokenLabel.trim() || `gpu-node-${enrollmentTokens.length + 1}`;
    const data = await issueEnrollmentToken(label, poolMode);
    if (data) {
      setIssuedToken(data);
      setHasCopiedIssuedToken(false);
      const readyCmd = normalizeHubCommand(data.dockerRunLinuxCommand, true);
      setSavedCmdByTokenId((prev) => {
        const next = { ...prev, [data.id]: readyCmd };
        try {
          window.sessionStorage.setItem("cutmesh_issued_token_cmds_v1", JSON.stringify(next));
        } catch { }
        return next;
      });
      onNotify(`Token ${data.tokenPrefix} generato e comando salvato`);
      onRefresh();
    }
  };

  const handlePrepareAnotherGpu = () => {
    setIssuedToken(null);
    const match = tokenLabel.match(/^(.*?)(\d+)$/);
    if (match) {
      const nextNum = String(Number(match[2]) + 1).padStart(match[2].length, "0");
      setTokenLabel(`${match[1]}${nextNum}`);
    } else {
      setTokenLabel(`${tokenLabel}-02`);
    }
  };

  const { locale } = useI18n();
  const isEn = locale === "en";

  const handleRevokeToken = async (tokenId: string) => {
    await revokeEnrollmentToken(tokenId);
    onNotify(isEn ? "Enrollment token revoked" : "Token di arruolamento revocato");
    onRefresh();
  };

  const handleTestWatchdog = async (worker: GpuWorkerNode) => {
    const ok = await triggerWatchdogFault(worker.id);
    if (ok) {
      onNotify(
        isEn
          ? `Recovery check triggered on ${worker.gpuName}: auto-restart in progress.`
          : `Verifica ripristino avviata su ${worker.gpuName}: riavvio automatico in corso.`
      );
      setTimeout(onRefresh, 2200);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold">{t.providerTitle}</h2>
          <p className="text-xs text-muted-foreground">{t.providerSubtitle}</p>
        </div>

        <Dialog open={gpuModalOpen} onOpenChange={setGpuModalOpen}>
          <DialogTrigger asChild>
            <Button className="gap-2">
              <Plus className="h-4 w-4" />
              {t.connectNewGpuBtn}
            </Button>
          </DialogTrigger>
          <DialogContent className="w-[95vw] max-w-2xl max-h-[90vh] overflow-y-auto overflow-x-hidden">
            <DialogHeader>
              <DialogTitle>{t.wizardTitle}</DialogTitle>
              <DialogDescription>{t.wizardDesc}</DialogDescription>
            </DialogHeader>

            <div className="space-y-4 py-2 min-w-0">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 min-w-0">
                <div className="space-y-1.5 min-w-0">
                  <Label className="text-xs">
                    {isEn ? "GPU Node Identifier" : "Nome Identificativo Nodo GPU"}
                  </Label>
                  <Input
                    value={tokenLabel}
                    onChange={(e) => setTokenLabel(e.target.value)}
                    placeholder={isEn ? "Node identifier" : "Nome identificativo nodo"}
                    className="h-9 text-xs"
                  />
                </div>
                <div className="space-y-1.5 min-w-0">
                  <Label className="text-xs">
                    {isEn ? "Pool Allocation Mode" : "Modalità Allocazione Pool"}
                  </Label>
                  <Select
                    value={poolMode}
                    onValueChange={(v) => setPoolMode(v as "SHARED_POOL" | "PRIVATE_SOLO")}
                  >
                    <SelectTrigger className="h-9 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="SHARED_POOL" className="text-xs">
                        {isEn
                          ? "SHARED_POOL — Shared Pool (Auto-Pilot)"
                          : "SHARED_POOL — Pool Condiviso (Auto-Pilot)"}
                      </SelectItem>
                      <SelectItem value="PRIVATE_SOLO" className="text-xs">
                        {isEn
                          ? "PRIVATE_SOLO — Exclusive Private Use"
                          : "PRIVATE_SOLO — Uso Privato Esclusivo"}
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="text-[11px] text-muted-foreground border border-border bg-muted/30 px-3 py-2 rounded-[2px] leading-relaxed">
                {poolMode === "SHARED_POOL" ? (
                  isEn ? (
                    <>
                      <strong className="text-foreground">SHARED_POOL:</strong> the GPU joins the global pool;{" "}
                      <span className="font-mono">AUTO_PILOT</span> automatically pairs it with other GPUs to form distributed Rigs and serve API traffic.
                    </>
                  ) : (
                    <>
                      <strong className="text-foreground">SHARED_POOL:</strong> la GPU entra nel pool globale;{" "}
                      <span className="font-mono">AUTO_PILOT</span> la abbina automaticamente con altre GPU per formare Rig distribuiti e servire il traffico API.
                    </>
                  )
                ) : isEn ? (
                  <>
                    <strong className="text-foreground">PRIVATE_SOLO:</strong> the GPU stays reserved exclusively for your account and is never shared with other tenants.
                  </>
                ) : (
                  <>
                    <strong className="text-foreground">PRIVATE_SOLO:</strong> la GPU resta riservata esclusivamente al tuo account e non viene mai condivisa con altri tenant.
                  </>
                )}
              </div>

              <div className="space-y-2 min-w-0">
                <div className="flex justify-between items-center text-xs">
                  <Label className="text-xs">
                    {isEn
                      ? "Allocated GPU Power Budget (--power-pct)"
                      : "Budget Potenza GPU Allocata (--power-pct)"}
                  </Label>
                  <Badge variant="outline" className="font-mono text-[10px]">
                    {powerPct}%
                  </Badge>
                </div>
                <Slider
                  min={20}
                  max={100}
                  step={5}
                  value={[powerPct]}
                  onValueChange={([val]) => setPowerPct(val)}
                />
              </div>

              <div className="flex items-center gap-2">
                <Button onClick={handleIssueToken} className="flex-1 gap-2">
                  {isEn ? "Generate Token & Startup Command" : "Genera Token & Comando di Avvio"}
                </Button>
                {issuedToken && (
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handlePrepareAnotherGpu}
                    className="gap-1.5 shrink-0"
                  >
                    <Plus className="h-3.5 w-3.5" />
                    {isEn ? "Add another GPU" : "Aggiungi altra GPU"}
                  </Button>
                )}
              </div>

              {issuedToken && (
                <Card className="bg-muted/30 min-w-0 overflow-hidden">
                  <CardHeader className="p-3.5 pb-2 space-y-2">
                    <div
                      className={`text-xs px-3 py-2 border rounded-[2px] flex items-center justify-between gap-2 ${
                        hasCopiedIssuedToken
                          ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-400"
                          : "border-primary/60 bg-primary/10 text-primary"
                      }`}
                    >
                      <span className="font-semibold">
                        {hasCopiedIssuedToken
                          ? isEn
                            ? "✓ Token / command copied — you can safely close this modal."
                            : "✓ Token / comando copiato — puoi chiudere in sicurezza."
                          : isEn
                          ? "⚠ Care: did you copy the token? Save it before closing — only its hash is stored on the server."
                          : "⚠ Attento: hai copiato il token? Salvalo prima di chiudere, sul server resta solo l'hash."}
                      </span>
                    </div>
                    <div className="flex flex-wrap items-center justify-between gap-2 min-w-0">
                      <span className="text-xs font-semibold">
                        {isEn ? "Generated Token:" : "Token Generato:"}
                      </span>
                      <Badge variant="outline" className="font-mono text-[10px] max-w-full truncate">
                        {issuedToken.rawTokenSecret}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="p-3.5 pt-2 min-w-0 overflow-hidden">
                    <Tabs defaultValue="linux" className="w-full min-w-0">
                      <TabsList className="grid grid-cols-3 w-full">
                        <TabsTrigger value="linux" className="text-xs">Linux / Cloud</TabsTrigger>
                        <TabsTrigger value="wsl2" className="text-xs">Windows (WSL2)</TabsTrigger>
                        <TabsTrigger value="rust" className="text-xs">{isEn ? "CLI Agent" : "Agente CLI"}</TabsTrigger>
                      </TabsList>
                      <TabsContent value="linux" className="space-y-2 min-w-0">
                        <div className="w-full overflow-x-auto">
                          <CodeBlock
                            language="bash"
                            value={normalizeHubCommand(issuedToken.dockerRunLinuxCommand, true)}
                          />
                        </div>
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => {
                              setHasCopiedIssuedToken(true);
                              onCopyText(
                                normalizeHubCommand(issuedToken.dockerRunLinuxCommand, true),
                                "cmd-linux"
                              );
                            }}
                          >
                            <Copy className="h-3.5 w-3.5 mr-1.5" />
                            {copiedId === "cmd-linux"
                              ? isEn
                                ? "Copied!"
                                : "Copiato!"
                              : isEn
                              ? "Copy Linux Docker Command"
                              : "Copia Comando Docker Linux"}
                          </Button>
                          <span className="text-[11px] text-muted-foreground font-mono">
                            Hub: {API_BASE} · UDP 40000
                          </span>
                        </div>
                      </TabsContent>
                      <TabsContent value="wsl2" className="space-y-2 min-w-0">
                        <div className="w-full overflow-x-auto">
                          <CodeBlock
                            language="powershell"
                            value={normalizeHubCommand(issuedToken.dockerRunWsl2Command, true)}
                          />
                        </div>
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => {
                              setHasCopiedIssuedToken(true);
                              onCopyText(
                                normalizeHubCommand(issuedToken.dockerRunWsl2Command, true),
                                "cmd-wsl2"
                              );
                            }}
                          >
                            <Copy className="h-3.5 w-3.5 mr-1.5" />
                            {copiedId === "cmd-wsl2"
                              ? isEn
                                ? "Copied!"
                                : "Copiato!"
                              : isEn
                              ? "Copy Windows / WSL2 Command"
                              : "Copia Comando Windows / WSL2"}
                          </Button>
                          <span className="text-[11px] text-muted-foreground font-mono">
                            Docker Desktop + NVIDIA GPU
                          </span>
                        </div>
                      </TabsContent>
                      <TabsContent value="rust" className="space-y-2 min-w-0">
                        <div className="w-full overflow-x-auto">
                          <CodeBlock
                            language="bash"
                            value={`${normalizeHubCommand(issuedToken.nativeRustCliCommand, false)} --power-pct ${powerPct}`}
                          />
                        </div>
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => {
                            setHasCopiedIssuedToken(true);
                            onCopyText(
                              `${normalizeHubCommand(issuedToken.nativeRustCliCommand, false)} --power-pct ${powerPct}`,
                              "cmd-rust"
                            );
                          }}
                        >
                          <Copy className="h-3.5 w-3.5 mr-1.5" />
                          {copiedId === "cmd-rust"
                            ? isEn
                              ? "Copied!"
                              : "Copiato!"
                            : isEn
                            ? "Copy Agent Command"
                            : "Copia Comando Agente"}
                        </Button>
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </Card>
              )}
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {/* Always-Clear 3-Step Worker Quickstart Guide */}
      <Card className="border-primary/25 bg-primary/[0.03]">
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <CardTitle className="text-sm flex items-center gap-2">
                <Terminal className="h-4 w-4 text-primary" />
                {isEn
                  ? "Quickstart: How to Launch a GPU Worker Node"
                  : "Guida Rapida: Come avviare un Nodo Worker GPU"}
              </CardTitle>
              <CardDescription className="text-xs">
                {isEn
                  ? "Connect any Linux server, Vast.ai instance, or Windows/WSL2 PC with an NVIDIA GPU in 3 steps."
                  : "Collega qualsiasi server Linux, Vast.ai o PC Windows/WSL2 con GPU NVIDIA in 3 passaggi."}
              </CardDescription>
            </div>
            <Badge variant="outline" className="font-mono text-[11px]">
              Hub: {API_BASE}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="rounded-lg border bg-card/70 p-3 space-y-1.5">
              <div className="flex items-center gap-2 text-xs font-semibold">
                <Badge className="h-5 w-5 p-0 justify-center rounded-full text-[10px]">1</Badge>
                <Cpu className="h-3.5 w-3.5 text-primary" />
                {isEn ? "Verify GPU & Docker" : "Verifica GPU & Docker"}
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                {isEn ? (
                  <>
                    Ensure NVIDIA drivers (&gt;= 550), Docker, and{" "}
                    <code className="font-mono text-foreground">nvidia-container-toolkit</code> are installed.
                    Open UDP port <code className="font-mono text-foreground">40000</code> for direct P2P links.
                  </>
                ) : (
                  <>
                    Assicurati di avere driver NVIDIA (&gt;= 550), Docker e{" "}
                    <code className="font-mono text-foreground">nvidia-container-toolkit</code> installati.
                    Apri la porta UDP <code className="font-mono text-foreground">40000</code> per connessione P2P diretta.
                  </>
                )}
              </p>
            </div>

            <div className="rounded-lg border bg-card/70 p-3 space-y-1.5">
              <div className="flex items-center gap-2 text-xs font-semibold">
                <Badge className="h-5 w-5 p-0 justify-center rounded-full text-[10px]">2</Badge>
                <ShieldCheck className="h-3.5 w-3.5 text-primary" />
                {isEn ? "Generate Node Token" : "Genera il Token del Nodo"}
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                {isEn ? (
                  <>
                    Click <strong>{t.connectNewGpuBtn}</strong> above, set the node name and power budget, and copy the ready-to-run command with URL and token pre-filled.
                  </>
                ) : (
                  <>
                    Clicca <strong>{t.connectNewGpuBtn}</strong> in alto a destra, scegli il nome del nodo e il budget potenza, e ottieni il comando pronto con URL e token già configurati.
                  </>
                )}
              </p>
            </div>

            <div className="rounded-lg border bg-card/70 p-3 space-y-1.5">
              <div className="flex items-center gap-2 text-xs font-semibold">
                <Badge className="h-5 w-5 p-0 justify-center rounded-full text-[10px]">3</Badge>
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                {isEn ? "Start & Verify Online" : "Avvia e Verifica Online"}
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                {isEn ? (
                  <>
                    Paste the <code className="font-mono text-foreground">docker run</code> command on the GPU host. Within 5 seconds the node appears below as{" "}
                    <strong className="text-emerald-500">ONLINE HEALTHY</strong>.
                  </>
                ) : (
                  <>
                    Incolla il comando <code className="font-mono text-foreground">docker run</code> sul server GPU. Entro 5 secondi il nodo apparirà qui sotto con stato{" "}
                    <strong className="text-emerald-500">ONLINE HEALTHY</strong>.
                  </>
                )}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Live Silicon GPU Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {workers.map((w) => {
          const vramUsedMb = Math.max(0, w.totalVramMb - w.freeVramMb);
          const usedPct = Math.min(
            100,
            Math.round((vramUsedMb / Math.max(1, w.totalVramMb)) * 100)
          );

          return (
            <Card key={w.id}>
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="flex items-center gap-2">
                      <CardTitle className="text-base">{w.gpuName}</CardTitle>
                      <Badge variant={w.rpcHealthy ? "default" : "destructive"}>
                        {w.rpcHealthy ? "ONLINE HEALTHY" : "OFFLINE"}
                      </Badge>
                    </div>
                    <CardDescription className="font-mono text-xs mt-1">
                      {isEn ? "Node" : "Nodo"}: <strong>{w.nodeName}</strong> · Device: {w.device}
                    </CardDescription>
                  </div>
                  <Badge
                    variant={
                      w.assignmentState === "LOCKED_IN_RIG" ? "secondary" : "outline"
                    }
                  >
                    {w.assignmentState}
                  </Badge>
                </div>
              </CardHeader>

              <CardContent className="space-y-4">
                <Card className="bg-muted/30 shadow-none overflow-hidden">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="h-7 text-[10px] text-center">Compute</TableHead>
                        <TableHead className="h-7 text-[10px] text-center">Bandwidth</TableHead>
                        <TableHead className="h-7 text-[10px] text-center">Power Budget</TableHead>
                        <TableHead className="h-7 text-[10px] text-center">Temp / Load</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <TableRow>
                        <TableCell className="py-2 text-center font-mono text-xs font-bold">
                          {w.effectiveTflops ? `${w.effectiveTflops.toFixed(1)} TF` : "—"}
                        </TableCell>
                        <TableCell className="py-2 text-center font-mono text-xs font-bold">
                          {w.effectiveBandwidthGbS ? `${w.effectiveBandwidthGbS.toFixed(0)} GB/s` : "—"}
                        </TableCell>
                        <TableCell className="py-2 text-center font-mono text-xs font-bold">
                          {w.allocatedPowerWatts ? `${w.allocatedPowerWatts.toFixed(0)}W` : "—"} ({w.powerBudgetPct ?? 100}%)
                        </TableCell>
                        <TableCell className="py-2 text-center font-mono text-xs font-bold">
                          {w.temperatureC != null ? `${w.temperatureC}°C` : "—"} • {w.gpuUtilizationPct != null ? `${w.gpuUtilizationPct}%` : "—"}
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </Card>

                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">
                      {isEn ? "VRAM Used" : "VRAM Usata"}:{" "}
                      <strong className="text-foreground">{vramUsedMb.toLocaleString()} MB</strong> /{" "}
                      {w.totalVramMb.toLocaleString()} MB ({usedPct}%)
                    </span>
                    <span className="font-mono text-muted-foreground">
                      {isEn ? "Free" : "Libera"}: <strong className="text-foreground">{w.freeVramMb.toLocaleString()} MB</strong>
                    </span>
                  </div>
                  <Progress value={usedPct} />
                </div>
              </CardContent>

              <CardFooter className="flex flex-wrap items-center justify-between gap-2 border-t pt-4">
                <div className="text-xs text-muted-foreground">
                  {isEn ? "Auto restarts" : "Ripristini automatici"}:{" "}
                  <strong className="text-foreground font-mono">
                    {w.watchdogRestarts}
                  </strong>
                </div>

                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => onCopyText(w.irohNodeId || w.id, `tk-${w.id}`)}
                  >
                    <Copy className="h-3.5 w-3.5 mr-1" />
                    {copiedId === `tk-${w.id}`
                      ? isEn
                        ? "ID Copied"
                        : "ID Copiato"
                      : isEn
                      ? "Copy Node ID"
                      : "Copia ID Nodo"}
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => handleTestWatchdog(w)}
                  >
                    {t.testWatchdogBtn}
                  </Button>
                </div>
              </CardFooter>
            </Card>
          );
        })}
      </div>

      {/* Enrollment Tokens Management Table */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-2 pb-3">
          <CardTitle className="text-sm">{t.tokensTableTitle}</CardTitle>
          {revokedTokensCount > 0 && (
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-[11px] font-mono"
              onClick={() => setShowRevokedTokens((v) => !v)}
            >
              {showRevokedTokens
                ? isEn
                  ? `Hide revoked (${revokedTokensCount})`
                  : `Nascondi revocati (${revokedTokensCount})`
                : isEn
                ? `Show revoked (${revokedTokensCount})`
                : `Mostra revocati (${revokedTokensCount})`}
            </Button>
          )}
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{isEn ? "Node Name" : "Nome Nodo"}</TableHead>
                <TableHead>{isEn ? "Token Prefix" : "Prefisso Token"}</TableHead>
                <TableHead>{isEn ? "Default Pool" : "Pool Default"}</TableHead>
                <TableHead>{isEn ? "Last Heartbeat" : "Ultimo Heartbeat"}</TableHead>
                <TableHead>{isEn ? "Status" : "Stato"}</TableHead>
                <TableHead></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {visibleTokens.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-6 text-xs text-muted-foreground">
                    {isEn ? (
                      <>
                        No active tokens. Click <strong>{t.connectNewGpuBtn}</strong> to create one.
                      </>
                    ) : (
                      <>
                        Nessun token attivo. Clicca <strong>{t.connectNewGpuBtn}</strong> per crearne uno.
                      </>
                    )}
                  </TableCell>
                </TableRow>
              ) : (
                visibleTokens.map((tk) => (
                  <TableRow key={tk.id} className={tk.revoked ? "opacity-60" : ""}>
                    <TableCell className="font-semibold">{tk.label}</TableCell>
                    <TableCell className="font-mono">{tk.tokenPrefix}</TableCell>
                    <TableCell>{tk.defaultPoolMode}</TableCell>
                    <TableCell className="text-muted-foreground font-mono text-xs">
                      {tk.lastUsedAt ? new Date(tk.lastUsedAt).toLocaleTimeString() : "—"}
                    </TableCell>
                    <TableCell>
                      <Badge variant={tk.revoked ? "destructive" : "default"}>
                        {tk.revoked ? "REVOKED" : "ACTIVE"}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      {!tk.revoked && (
                        <div className="inline-flex items-center gap-1.5">
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-7 text-[11px] font-mono gap-1.5"
                            onClick={() => {
                              const cmd = savedCmdByTokenId[tk.id] || buildFallbackDockerCmd(tk);
                              onCopyText(cmd, `row-cmd-${tk.id}`);
                            }}
                          >
                            <Copy className="h-3 w-3" />
                            {copiedId === `row-cmd-${tk.id}`
                              ? isEn
                                ? "Copied!"
                                : "Copiato!"
                              : savedCmdByTokenId[tk.id]
                                ? isEn
                                  ? "Copy docker run"
                                  : "Copia docker run"
                                : isEn
                                  ? "Copy template"
                                  : "Copia template"}
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleRevokeToken(tk.id)}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      )}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}

