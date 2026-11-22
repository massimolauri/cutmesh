"use client";

import React, { createContext, useContext, useEffect, useState } from "react";

export type Locale = "it" | "en";

export const dictionaries = {
  it: {
    brandSubtitle: "Piattaforma AI Distribuita & Gestione Cluster GPU",
    signInWithGoogle: "Accedi",
    signedInAs: "Sessione attiva",
    roleAdmin: "Amministratore",
    roleProvider: "Fornitore GPU",
    roleDeveloper: "Sviluppatore API",
    activeRigsUnit: "RIG ATTIVI",
    lightTheme: "Tema Chiaro",
    darkTheme: "Tema Scuro",
    languageLabel: "Lingua",
    signOut: "Esci",
    // Sidebar Groups & Items
    groupCluster: "PIATTAFORMA",
    groupResources: "RISORSE & API",
    navOverview: "Panoramica Cluster",
    navOrchestrator: "Catalogo Modelli AI",
    navUsers: "Utenti & Permessi",
    groupProvider: "INFRASTRUTTURA GPU",
    navProvider: "Nodi GPU & Connessione",
    groupDeveloper: "SVILUPPATORI & API",
    navPlayground: "Playground Chat",
    navApiKeys: "Chiavi API & SDK",
    groupSecurity: "ATTIVITÀ & SICUREZZA",
    navAudit: "Registro Attività",
    // Overview
    kpiOnlineGpus: "Nodi GPU Online",
    kpiStandbyPool: "Pool Disponibile",
    kpiActiveRigs: "Cluster Attivi",
    kpiTotalVram: "Memoria VRAM Totale",
    kpiComputePower: "Capacità di Calcolo",
    pipelineTitle: "Topologia Cluster Attivi",
    pipelineSubtitle:
      "Distribuzione in tempo reale dei modelli sulle GPU connesse e instradamento delle richieste API.",
    recentAuditTitle: "Attività Recenti del Sistema",
    // Provider
    providerTitle: "Infrastruttura GPU & Connessione Nodi",
    providerSubtitle:
      "Collega server e GPU dedicate o cloud tramite connessione cifrata con monitoraggio continuo.",
    connectNewGpuBtn: "Collega Nuovo Nodo GPU",
    wizardTitle: "Collega un Nuovo Nodo GPU",
    wizardDesc:
      "Genera un token di accesso per associare un server GPU al cluster e copia il comando di avvio.",
    testWatchdogBtn: "Verifica Ripristino Nodo",
    tokensTableTitle: "Token di Accesso Nodi GPU",
    // Orchestrator
    orchestratorTitle: "Catalogo Modelli & Allocazione Cluster",
    orchestratorSubtitle:
      "Gestisci la distribuzione automatica o configura manualmente i cluster per ogni modello.",
    addModelBtn: "Aggiungi Modello",
    compileModalBtn: "Ottimizza Modello...",
    manualComposerTitle: "Configuratore Cluster Manuale",
    manualComposerDesc:
      "Seleziona il modello e le GPU disponibili per stimare allocazione e finestra di contesto.",
    // Playground
    playgroundTitle: "Playground Chat AI (/v1/chat/completions)",
    playgroundSubtitle:
      "Invia richieste in tempo reale ai cluster attivi attraverso l'endpoint compatibile OpenAI.",
    inspectorTitle: "Parametri & Telemetria",
    systemPromptLabel: "System Prompt",
    // API Keys
    apiKeysTitle: "Chiavi API (/v1) & Esempi SDK",
    apiKeysSubtitle:
      "Autentica applicazioni e client compatibili OpenAI verso l'endpoint /v1/chat/completions.",
    // Audit
    auditTitle: "Registro Eventi & Verifiche di Integrità",
    auditSubtitle:
      "Storico operativo delle allocazioni cluster, dello stato dei nodi e delle verifiche di sistema.",
    runAuditBtn: "Esegui Verifica Integrità",
    viewJsonBtn: "Dettagli JSON"
  },
  en: {
    brandSubtitle: "Distributed AI Cloud & GPU Cluster Console",
    signInWithGoogle: "Sign in",
    signedInAs: "Signed in",
    roleAdmin: "Administrator",
    roleProvider: "GPU Provider",
    roleDeveloper: "API Developer",
    activeRigsUnit: "ACTIVE RIGS",
    lightTheme: "Light Theme",
    darkTheme: "Dark Theme",
    languageLabel: "Language",
    signOut: "Sign out",
    // Sidebar Groups & Items
    groupCluster: "PLATFORM",
    groupResources: "RESOURCES & API",
    navOverview: "Cluster Overview",
    navOrchestrator: "Model Catalog",
    navUsers: "Users & Roles",
    groupProvider: "GPU INFRASTRUCTURE",
    navProvider: "GPU Nodes & Onboarding",
    groupDeveloper: "DEVELOPERS & API",
    navPlayground: "AI Playground",
    navApiKeys: "API Keys & SDK",
    groupSecurity: "ACTIVITY & SECURITY",
    navAudit: "Activity Log",
    // Overview
    kpiOnlineGpus: "Online GPU Nodes",
    kpiStandbyPool: "Available Pool",
    kpiActiveRigs: "Active Clusters",
    kpiTotalVram: "Total Fleet VRAM",
    kpiComputePower: "Compute Capacity",
    pipelineTitle: "Active Cluster Topology",
    pipelineSubtitle:
      "Real-time model placement across connected GPUs and API request routing.",
    recentAuditTitle: "Recent System Activity",
    // Provider
    providerTitle: "GPU Infrastructure & Node Onboarding",
    providerSubtitle:
      "Connect dedicated or cloud GPU servers over encrypted links with automated health monitoring.",
    connectNewGpuBtn: "Connect GPU Node",
    wizardTitle: "Connect a New GPU Node",
    wizardDesc:
      "Generate an enrollment token for a GPU server and copy the startup command.",
    testWatchdogBtn: "Test Node Recovery",
    tokensTableTitle: "GPU Enrollment Tokens",
    // Orchestrator
    orchestratorTitle: "Model Catalog & Cluster Orchestration",
    orchestratorSubtitle:
      "Manage automated scaling or compose clusters manually from available GPU nodes.",
    addModelBtn: "Add Model",
    compileModalBtn: "Optimize Model...",
    manualComposerTitle: "Manual Cluster Composer",
    manualComposerDesc:
      "Select a model and available GPUs to preview allocation and context capacity.",
    // Playground
    playgroundTitle: "AI Chat Playground (/v1/chat/completions)",
    playgroundSubtitle:
      "Send live inference requests to active clusters through the OpenAI-compatible gateway.",
    inspectorTitle: "Parameters & Telemetry",
    systemPromptLabel: "System Prompt",
    // API Keys
    apiKeysTitle: "API Keys (/v1) & SDK Quickstart",
    apiKeysSubtitle:
      "Authenticate OpenAI-compatible clients against the /v1/chat/completions endpoint.",
    // Audit
    auditTitle: "Activity Log & Integrity Checks",
    auditSubtitle:
      "Operational log of cluster allocations, node health, and system integrity checks.",
    runAuditBtn: "Run Integrity Check",
    viewJsonBtn: "Inspect JSON"
  }
};

interface I18nContextValue {
  locale: Locale;
  setLocale: (loc: Locale) => void;
  t: (typeof dictionaries)["it"];
}

const I18nContext = createContext<I18nContextValue>({
  locale: "it",
  setLocale: () => {},
  t: dictionaries.it
});

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("it");

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem("cutmesh_locale");
      if (saved === "en" || saved === "it") {
        setLocaleState(saved);
      }
    } catch {
      // ignore storage errors
    }
  }, []);

  const setLocale = (next: Locale) => {
    setLocaleState(next);
    try {
      window.localStorage.setItem("cutmesh_locale", next);
    } catch {
      // ignore storage errors
    }
  };

  return React.createElement(
    I18nContext.Provider,
    { value: { locale, setLocale, t: dictionaries[locale] } },
    children
  );
}

export function useI18n() {
  return useContext(I18nContext);
}
