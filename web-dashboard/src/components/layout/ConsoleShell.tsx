"use client";

import React from "react";
import { usePathname } from "next/navigation";
import { useConsole } from "@/context/ConsoleContext";
import { TopHeader } from "@/components/layout/TopHeader";
import { SidebarNav } from "@/components/layout/SidebarNav";
import { CutmeshLoader } from "@/components/ui/cutmesh-logo";
import { Button } from "@/components/ui/button";
import { TooltipProvider } from "@/components/ui/tooltip";
import { CheckCircle2 } from "lucide-react";

export function ConsoleShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname() || "/";
  const {
    locale,
    setLocale,
    t,
    authReady,
    user,
    activeRole,
    activeSection,
    navigateToSection,
    onlineWorkers,
    activeRigs,
    apiKeys,
    allUsers,
    auditEvents,
    toastMessage,
    setToastMessage,
    handleLogout
  } = useConsole();

  // Never flash /login on F5: show the animated Cutmesh laser-cut loader while checking session
  if (!authReady) {
    return <CutmeshLoader />;
  }

  // Dedicated unauthenticated /login route
  if (pathname === "/login") {
    if (user) {
      return <CutmeshLoader />;
    }
    return <>{children}</>;
  }

  // Redirecting unauthenticated visitor to /login (no flash of empty shell)
  if (!user) {
    return <CutmeshLoader />;
  }

  const totalFleetTflops = onlineWorkers.reduce(
    (acc, w) => acc + (w.effectiveTflops || 0),
    0
  );

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-background text-foreground flex flex-col">
        <TopHeader
          t={t}
          locale={locale}
          onLocaleSwitch={setLocale}
          activeRole={activeRole}
          onRoleSwitch={() => {}}
          user={user}
          onLogout={handleLogout}
          workersCount={onlineWorkers.length}
          totalFleetTflops={totalFleetTflops}
          activeRigsCount={activeRigs.length}
        />

        <div className="flex-1 flex overflow-hidden">
          <SidebarNav
            t={t}
            activeRole={activeRole}
            activeSection={activeSection}
            onSelectSection={navigateToSection}
            activeRigsCount={activeRigs.length}
            workersCount={onlineWorkers.length}
            activeApiKeysCount={apiKeys.filter((k) => !k.revoked).length}
            usersCount={allUsers.length}
            auditEventsCount={auditEvents.length}
          />

          <main className="flex-1 overflow-y-auto p-6 space-y-6">{children}</main>
        </div>

        {/* Floating Fixed Bottom-Right Swiss Toast (Zero Layout Shift) */}
        {toastMessage && (
          <div className="fixed bottom-5 right-5 z-50 flex items-center gap-3 border border-border border-l-2 border-l-primary bg-card text-foreground px-4 py-2.5 rounded-[2px] max-w-md shadow-xl">
            <CheckCircle2 className="h-4 w-4 text-primary shrink-0" />
            <span className="text-xs font-medium leading-snug">{toastMessage}</span>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-[11px] font-mono text-muted-foreground hover:text-foreground shrink-0"
              onClick={() => setToastMessage(null)}
            >
              ✕
            </Button>
          </div>
        )}
      </div>
    </TooltipProvider>
  );
}
