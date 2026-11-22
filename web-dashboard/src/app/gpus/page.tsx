"use client";

import React, { useEffect } from "react";
import { useRouter } from "next/navigation";
import { defaultPathForRole, useConsole } from "@/context/ConsoleContext";
import { GpuProviderSection } from "@/components/sections/GpuProviderSection";

export default function GpusPage() {
  const router = useRouter();
  const {
    t,
    activeRole,
    workers,
    enrollmentTokens,
    copiedId,
    copyText,
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
    <GpuProviderSection
      t={t}
      workers={workers}
      enrollmentTokens={enrollmentTokens}
      copiedId={copiedId}
      onCopyText={copyText}
      onNotify={notify}
      onRefresh={fetchAllData}
    />
  );
}
