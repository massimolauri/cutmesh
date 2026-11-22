"use client";

import React, { useEffect } from "react";
import { useRouter } from "next/navigation";
import { defaultPathForRole, useConsole } from "@/context/ConsoleContext";
import { SecurityAuditSection } from "@/components/sections/SecurityAuditSection";

export default function AuditPage() {
  const router = useRouter();
  const {
    t,
    locale,
    activeRole,
    auditEvents,
    auditCategoryFilter,
    setAuditCategoryFilter,
    notify,
    fetchAllData
  } = useConsole();

  useEffect(() => {
    if (activeRole !== "SUPER_ADMIN") {
      router.replace(defaultPathForRole(activeRole));
    }
  }, [activeRole, router]);

  if (activeRole !== "SUPER_ADMIN") {
    return null;
  }

  return (
    <SecurityAuditSection
      t={t}
      locale={locale}
      auditEvents={auditEvents}
      auditCategoryFilter={auditCategoryFilter}
      onChangeAuditCategoryFilter={setAuditCategoryFilter}
      onNotify={notify}
      onRefresh={fetchAllData}
    />
  );
}
