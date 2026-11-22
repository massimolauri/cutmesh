"use client";

import React, { useEffect } from "react";
import { useRouter } from "next/navigation";
import { defaultPathForRole, useConsole } from "@/context/ConsoleContext";
import { AiPlaygroundSection } from "@/components/sections/AiPlaygroundSection";

export default function PlaygroundPage() {
  const router = useRouter();
  const {
    t,
    activeRole,
    models,
    selectedPlaygroundModelId,
    latestRawApiKey,
    setLatestRawApiKey,
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
    <AiPlaygroundSection
      t={t}
      models={models}
      initialModelId={selectedPlaygroundModelId}
      latestRawApiKey={latestRawApiKey}
      onSetLatestRawApiKey={setLatestRawApiKey}
      onRefresh={fetchAllData}
    />
  );
}
