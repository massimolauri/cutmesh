"use client";

import React, { useEffect } from "react";
import { useRouter } from "next/navigation";
import { defaultPathForRole, useConsole } from "@/context/ConsoleContext";
import { ApiKeysSdkSection } from "@/components/sections/ApiKeysSdkSection";

export default function ApiKeysPage() {
  const router = useRouter();
  const {
    t,
    activeRole,
    apiKeys,
    latestRawApiKey,
    setLatestRawApiKey,
    copiedId,
    copyText,
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
    <ApiKeysSdkSection
      t={t}
      apiKeys={apiKeys}
      latestRawApiKey={latestRawApiKey}
      onSetLatestRawApiKey={setLatestRawApiKey}
      copiedId={copiedId}
      onCopyText={copyText}
      onNotify={notify}
      onRefresh={fetchAllData}
    />
  );
}
