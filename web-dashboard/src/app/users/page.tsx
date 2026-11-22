"use client";

import React, { useEffect } from "react";
import { useRouter } from "next/navigation";
import { defaultPathForRole, useConsole } from "@/context/ConsoleContext";
import { UsersRolesSection } from "@/components/sections/UsersRolesSection";
import {
  deleteTargetUser,
  registerUserAccount,
  updateTargetUserRole
} from "@/services/control-plane-api";

export default function UsersPage() {
  const router = useRouter();
  const {
    activeRole,
    setActiveRole,
    user,
    allUsers,
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
    <UsersRolesSection
      users={allUsers}
      currentUserId={user?.id}
      onUpdateUserRole={async (targetUserId, nextRole) => {
        await updateTargetUserRole(targetUserId, nextRole);
        if (targetUserId === user?.id) {
          setActiveRole(nextRole);
        }
        notify(`Ruolo utente aggiornato a ${nextRole}`);
        await fetchAllData();
      }}
      onCreateUser={async (payload) => {
        const res = await registerUserAccount({
          ...payload,
          preferredLocale: "it"
        });
        if (res.user) {
          notify(`Creato utente ${res.user.email} (${res.user.role})`);
          await fetchAllData();
        } else {
          notify(`Errore: ${res.error}`);
        }
      }}
      onDeleteUser={async (targetUserId) => {
        await deleteTargetUser(targetUserId);
        notify("Utente rimosso");
        await fetchAllData();
      }}
    />
  );
}
