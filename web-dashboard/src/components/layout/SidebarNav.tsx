"use client";

import React from "react";
import { ActiveSection, AppRole } from "@/types/control-plane";
import { Badge } from "@/components/ui/badge";
import { CutmeshWordmark } from "@/components/ui/cutmesh-logo";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Activity,
  Cpu,
  KeyRound,
  Layers,
  LayoutDashboard,
  MessageSquare,
  Users
} from "lucide-react";

interface SidebarNavProps {
  t: Record<string, string>;
  activeRole: AppRole;
  activeSection: ActiveSection;
  onSelectSection: (section: ActiveSection) => void;
  activeRigsCount: number;
  workersCount: number;
  activeApiKeysCount: number;
  usersCount: number;
  auditEventsCount: number;
}

export function SidebarNav({
  t,
  activeRole,
  activeSection,
  onSelectSection,
  activeRigsCount,
  workersCount,
  activeApiKeysCount,
  usersCount,
  auditEventsCount
}: SidebarNavProps) {
  return (
    <aside className="w-60 border-r bg-card/40 p-3 flex flex-col justify-between shrink-0">
      <div className="space-y-5">
        {activeRole === "SUPER_ADMIN" && (
          <>
            <div className="space-y-1">
              <div className="px-2.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                {t.groupCluster}
              </div>
              <Button
                variant={activeSection === "overview" ? "secondary" : "ghost"}
                className="w-full justify-start gap-2.5 text-xs"
                onClick={() => onSelectSection("overview")}
              >
                <LayoutDashboard className="h-4 w-4" />
                {t.navOverview}
              </Button>

              <Button
                variant={activeSection === "orchestrator" ? "secondary" : "ghost"}
                className="w-full justify-between text-xs"
                onClick={() => onSelectSection("orchestrator")}
              >
                <span className="flex items-center gap-2.5">
                  <Layers className="h-4 w-4" />
                  {t.navOrchestrator}
                </span>
                <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                  {activeRigsCount}
                </Badge>
              </Button>

              <Button
                variant={activeSection === "users" ? "secondary" : "ghost"}
                className="w-full justify-between text-xs"
                onClick={() => onSelectSection("users")}
              >
                <span className="flex items-center gap-2.5">
                  <Users className="h-4 w-4" />
                  {t.navUsers}
                </span>
                <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                  {usersCount}
                </Badge>
              </Button>

              <Button
                variant={activeSection === "audit" ? "secondary" : "ghost"}
                className="w-full justify-between text-xs"
                onClick={() => onSelectSection("audit")}
              >
                <span className="flex items-center gap-2.5">
                  <Activity className="h-4 w-4" />
                  {t.navAudit}
                </span>
                <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                  {auditEventsCount}
                </Badge>
              </Button>
            </div>

            <Separator />

            <div className="space-y-1">
              <div className="px-2.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                {t.groupResources}
              </div>
              <Button
                variant={activeSection === "provider" ? "secondary" : "ghost"}
                className="w-full justify-between text-xs"
                onClick={() => onSelectSection("provider")}
              >
                <span className="flex items-center gap-2.5">
                  <Cpu className="h-4 w-4" />
                  {t.navProvider}
                </span>
                <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                  {workersCount}
                </Badge>
              </Button>

              <Button
                variant={activeSection === "apikeys" ? "secondary" : "ghost"}
                className="w-full justify-between text-xs"
                onClick={() => onSelectSection("apikeys")}
              >
                <span className="flex items-center gap-2.5">
                  <KeyRound className="h-4 w-4" />
                  {t.navApiKeys}
                </span>
                <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                  {activeApiKeysCount}
                </Badge>
              </Button>

              <Button
                variant={activeSection === "playground" ? "secondary" : "ghost"}
                className="w-full justify-start gap-2.5 text-xs"
                onClick={() => onSelectSection("playground")}
              >
                <MessageSquare className="h-4 w-4" />
                {t.navPlayground}
              </Button>
            </div>
          </>
        )}

        {activeRole === "HARDWARE_PROVIDER" && (
          <div className="space-y-1">
            <div className="px-2.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              {t.groupProvider}
            </div>
            <Button
              variant={activeSection === "provider" ? "secondary" : "ghost"}
              className="w-full justify-between text-xs"
              onClick={() => onSelectSection("provider")}
            >
              <span className="flex items-center gap-2.5">
                <Cpu className="h-4 w-4" />
                {t.navProvider}
              </span>
              <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                {workersCount}
              </Badge>
            </Button>

            <Button
              variant={activeSection === "overview" ? "secondary" : "ghost"}
              className="w-full justify-start gap-2.5 text-xs"
              onClick={() => onSelectSection("overview")}
            >
              <LayoutDashboard className="h-4 w-4" />
              {t.navOverview}
            </Button>
          </div>
        )}

        {activeRole === "API_USER" && (
          <div className="space-y-1">
            <div className="px-2.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              {t.groupDeveloper}
            </div>
            <Button
              variant={activeSection === "orchestrator" ? "secondary" : "ghost"}
              className="w-full justify-between text-xs"
              onClick={() => onSelectSection("orchestrator")}
            >
              <span className="flex items-center gap-2.5">
                <Layers className="h-4 w-4" />
                {t.navOrchestrator}
              </span>
              <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                {activeRigsCount}
              </Badge>
            </Button>

            <Button
              variant={activeSection === "playground" ? "secondary" : "ghost"}
              className="w-full justify-start gap-2.5 text-xs"
              onClick={() => onSelectSection("playground")}
            >
              <MessageSquare className="h-4 w-4" />
              {t.navPlayground}
            </Button>

            <Button
              variant={activeSection === "apikeys" ? "secondary" : "ghost"}
              className="w-full justify-between text-xs"
              onClick={() => onSelectSection("apikeys")}
            >
              <span className="flex items-center gap-2.5">
                <KeyRound className="h-4 w-4" />
                {t.navApiKeys}
              </span>
              <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                {activeApiKeysCount}
              </Badge>
            </Button>
          </div>
        )}
      </div>

      <div className="px-2.5 pt-3 border-t border-border flex items-center justify-between text-[11px] font-mono text-muted-foreground">
        <CutmeshWordmark className="text-[11px]" iconSize={18} />
        <span className="text-[10px] text-primary">v2.0</span>
      </div>
    </aside>
  );
}
