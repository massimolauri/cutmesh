"use client";

import React, { useEffect } from "react";
import { useRouter } from "next/navigation";
import { defaultPathForRole, useConsole } from "@/context/ConsoleContext";
import { CatalogOrchestratorSection } from "@/components/sections/CatalogOrchestratorSection";

export default function CatalogPage() {
  const router = useRouter();
  const {
    t,
    activeRole,
    models,
    standbyWorkers,
    activeRigs,
    setSelectedPlaygroundModelId,
    navigateToSection,
    notify,
    fetchAllData
  } = useConsole();

  useEffect(() => {
    if (activeRole !== "SUPER_ADMIN" && activeRole !== "API_USER") {
      router.replace(defaultPathForRole(activeRole));
    }
  }, [activeRole, router]);

  if (activeRole !== "SUPER_ADMIN" && activeRole !== "API_USER") {
    return null;
  }

  return (
    <CatalogOrchestratorSection
      t={t}
      models={models}
      standbyWorkers={standbyWorkers}
      activeRigs={activeRigs}
      isAdmin={activeRole === "SUPER_ADMIN"}
      onSelectModelForPlayground={(modelId) => {
        setSelectedPlaygroundModelId(modelId);
        navigateToSection("playground");
      }}
      onNavigateToApiKeys={() => navigateToSection("apikeys")}
      onNotify={notify}
      onRefresh={fetchAllData}
    />
  );
}
