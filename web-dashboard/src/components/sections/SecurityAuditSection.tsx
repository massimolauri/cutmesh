"use client";

import React, { useState } from "react";
import { Locale } from "@/i18n/translations";
import { SecurityAuditEvent } from "@/types/control-plane";
import { runLiveCovariantAudit } from "@/services/control-plane-api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import { CodeBlock } from "@/components/ui/chat-markdown";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";
import { FileCode2, Play } from "lucide-react";

interface SecurityAuditSectionProps {
  t: Record<string, string>;
  locale: Locale;
  auditEvents: SecurityAuditEvent[];
  auditCategoryFilter: string;
  onChangeAuditCategoryFilter: (cat: string) => void;
  onNotify: (msg: string) => void;
  onRefresh: () => void;
}

export function SecurityAuditSection({
  t,
  locale,
  auditEvents,
  auditCategoryFilter,
  onChangeAuditCategoryFilter,
  onNotify,
  onRefresh
}: SecurityAuditSectionProps) {
  const [runningAudit, setRunningAudit] = useState<boolean>(false);
  const [inspectedAuditEvent, setInspectedAuditEvent] =
    useState<SecurityAuditEvent | null>(null);

  const isEn = locale === "en";

  const handleRunLiveAudit = async () => {
    setRunningAudit(true);
    try {
      const rep = await runLiveCovariantAudit(896);
      if (rep) {
        onNotify(
          isEn
            ? `Integrity check completed in ${rep.elapsedMs} ms`
            : `Verifica di integrità completata in ${rep.elapsedMs} ms`
        );
        onRefresh();
      }
    } finally {
      setRunningAudit(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold">{t.auditTitle}</h2>
          <p className="text-xs text-muted-foreground">{t.auditSubtitle}</p>
        </div>

        <Button
          onClick={handleRunLiveAudit}
          disabled={runningAudit}
          variant="outline"
          className="gap-2"
        >
          <Play className="h-4 w-4" />
          {runningAudit
            ? isEn
              ? "Checking..."
              : "Verifica in corso..."
            : t.runAuditBtn}
        </Button>
      </div>

      {/* Security & Operational Audit Event Log */}
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <CardTitle className="text-sm">
              {isEn ? "Event History" : "Storico Eventi"}
            </CardTitle>
            <Select
              value={auditCategoryFilter}
              onValueChange={onChangeAuditCategoryFilter}
            >
              <SelectTrigger className="w-64">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="ALL">
                  {isEn ? `All Events (${auditEvents.length})` : `Tutti gli Eventi (${auditEvents.length})`}
                </SelectItem>
                <SelectItem value="RIG_CLOSED_AND_LOCKED">
                  {isEn ? "Cluster Allocation" : "Allocazione Cluster"}
                </SelectItem>
                <SelectItem value="WORKER_ENROLLMENT_TOKEN">
                  {isEn ? "Tokens & GPU Nodes" : "Token & Nodi GPU"}
                </SelectItem>
                <SelectItem value="SELF_HEALING_WATCHDOG">
                  {isEn ? "GPU Auto-Recovery" : "Ripristino Automatico GPU"}
                </SelectItem>
                <SelectItem value="COVARIANT_PROTOCOL_AUDIT">
                  {isEn ? "Integrity Checks" : "Verifiche Integrità"}
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{isEn ? "Time" : "Ora"}</TableHead>
                <TableHead>{isEn ? "Category" : "Categoria"}</TableHead>
                <TableHead>{isEn ? "Status" : "Stato"}</TableHead>
                <TableHead>{isEn ? "Event Detail" : "Dettaglio Evento"}</TableHead>
                <TableHead></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {auditEvents.map((ev) => (
                <TableRow key={ev.id}>
                  <TableCell className="font-mono text-xs text-muted-foreground whitespace-nowrap">
                    {new Date(ev.createdAt).toLocaleTimeString()}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline" className="font-mono text-[10px]">
                      {ev.category}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant={ev.passed ? "secondary" : "destructive"}>
                      {ev.severity}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="font-semibold text-xs">
                      {locale === "it" ? ev.titleIt : ev.titleEn}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {locale === "it" ? ev.summaryIt : ev.summaryEn}
                    </div>
                  </TableCell>
                  <TableCell>
                    {ev.metricsJson && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setInspectedAuditEvent(ev)}
                      >
                        <FileCode2 className="h-3.5 w-3.5 mr-1" />
                        {t.viewJsonBtn}
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Dialog
        open={!!inspectedAuditEvent}
        onOpenChange={(o) => !o && setInspectedAuditEvent(null)}
      >
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>{isEn ? "Event Details" : "Dettagli Evento"}</DialogTitle>
            <DialogDescription>
              {isEn
                ? "Recorded payload for this system event."
                : "Dati registrati per questo evento di sistema."}
            </DialogDescription>
          </DialogHeader>
          <ScrollArea className="max-h-[420px]">
            <CodeBlock
              language="json"
              className="my-0"
              value={
                inspectedAuditEvent?.metricsJson
                  ? (() => {
                      try {
                        return JSON.stringify(
                          JSON.parse(inspectedAuditEvent.metricsJson),
                          null,
                          2
                        );
                      } catch {
                        return inspectedAuditEvent.metricsJson;
                      }
                    })()
                  : "{}"
              }
            />
          </ScrollArea>
        </DialogContent>
      </Dialog>
    </div>
  );
}
