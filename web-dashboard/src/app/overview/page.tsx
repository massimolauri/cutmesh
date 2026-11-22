"use client";

import React, { useEffect } from "react";
import { useRouter } from "next/navigation";
import { defaultPathForRole, useConsole } from "@/context/ConsoleContext";
import { ClusterOverviewSection } from "@/components/sections/ClusterOverviewSection";
import {
  createManualRig,
  deleteWorkerNode,
  disconnectWorkerNode,
  dissolveManagedRig,
  triggerWatchdogFault
} from "@/services/control-plane-api";

export default function OverviewPage() {
  const router = useRouter();
  const {
    t,
    activeRole,
    workers,
    activeRigs,
    models,
    navigateToSection,
    notify,
    fetchAllData
  } = useConsole();

  useEffect(() => {
    if (activeRole !== "SUPER_ADMIN" && activeRole !== "HARDWARE_PROVIDER") {
      router.replace(defaultPathForRole(activeRole));
    }
  }, [activeRole, router]);

  if (activeRole !== "SUPER_ADMIN" && activeRole !== "HARDWARE_PROVIDER") {
    return null;
  }

  return (
    <ClusterOverviewSection
      t={t}
      workers={workers}
      activeRigs={activeRigs}
      models={models}
      onNavigate={navigateToSection}
      onTriggerWatchdog={async (workerId, gpuName) => {
        await triggerWatchdogFault(workerId);
        notify(`Verifica ripristino inviata a ${gpuName}`);
        await fetchAllData();
      }}
      onDisconnectWorker={async (workerId, nodeName) => {
        await disconnectWorkerNode(workerId);
        notify(`GPU '${nodeName}' scollegata: topologia cluster aggiornata`);
        await fetchAllData();
      }}
      onDeleteWorker={async (workerId, nodeName) => {
        await deleteWorkerNode(workerId);
        notify(`Nodo offline '${nodeName}' rimosso`);
        await fetchAllData();
      }}
      onDissolveRig={async (rigId) => {
        await dissolveManagedRig(rigId);
        notify("Cluster terminato: GPU disponibili nel pool");
        await fetchAllData();
      }}
      onCreateOrReconfigureRig={async (rigName, modelId, workerIds) => {
        const res = await createManualRig(rigName, modelId, workerIds);
        if (res.ok) {
          notify(`Cluster '${rigName}' (${workerIds.length} GPU) attivato con successo!`);
        } else {
          notify(`Errore attivazione cluster: ${res.errorMessage}`);
        }
        await fetchAllData();
      }}
    />
  );
}
