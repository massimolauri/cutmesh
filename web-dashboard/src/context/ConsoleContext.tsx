"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Locale, useI18n } from "@/i18n/translations";
import {
  ActiveSection,
  AppRole,
  CatalogModel,
  EnrollmentTokenItem,
  GpuWorkerNode,
  ManagedRig,
  OpenAiApiKeyItem,
  SecurityAuditEvent,
  UserProfile
} from "@/types/control-plane";
import {
  clearAuthSession,
  fetchControlPlaneSnapshot,
  getStoredJwtToken
} from "@/services/control-plane-api";

export const SECTION_TO_PATH: Record<ActiveSection, string> = {
  overview: "/overview",
  orchestrator: "/catalog",
  provider: "/gpus",
  apikeys: "/api-keys",
  playground: "/playground",
  users: "/users",
  audit: "/audit"
};

export function pathToSection(pathname: string): ActiveSection {
  if (pathname.startsWith("/catalog")) return "orchestrator";
  if (pathname.startsWith("/gpus")) return "provider";
  if (pathname.startsWith("/api-keys")) return "apikeys";
  if (pathname.startsWith("/playground")) return "playground";
  if (pathname.startsWith("/users")) return "users";
  if (pathname.startsWith("/audit")) return "audit";
  return "overview";
}

export function defaultPathForRole(role: AppRole, hasIssuedApiKey?: boolean): string {
  if (role === "SUPER_ADMIN") return "/overview";
  if (role === "HARDWARE_PROVIDER") return "/gpus";
  return hasIssuedApiKey ? "/api-keys" : "/catalog";
}

interface ConsoleContextValue {
  locale: Locale;
  setLocale: (l: Locale) => void;
  t: Record<string, string>;
  authReady: boolean;
  user: UserProfile | null;
  activeRole: AppRole;
  setActiveRole: (role: AppRole) => void;
  activeSection: ActiveSection;
  navigateToSection: (section: ActiveSection) => void;
  allUsers: UserProfile[];
  workers: GpuWorkerNode[];
  onlineWorkers: GpuWorkerNode[];
  standbyWorkers: GpuWorkerNode[];
  enrollmentTokens: EnrollmentTokenItem[];
  models: CatalogModel[];
  rigs: ManagedRig[];
  activeRigs: ManagedRig[];
  auditEvents: SecurityAuditEvent[];
  auditCategoryFilter: string;
  setAuditCategoryFilter: (cat: string) => void;
  apiKeys: OpenAiApiKeyItem[];
  latestRawApiKey: string;
  setLatestRawApiKey: (k: string) => void;
  selectedPlaygroundModelId: string;
  setSelectedPlaygroundModelId: (id: string) => void;
  toastMessage: string | null;
  setToastMessage: (msg: string | null) => void;
  copiedId: string | null;
  notify: (msg: string) => void;
  copyText: (text: string, id: string) => void;
  fetchAllData: () => Promise<void>;
  handleAuthenticated: (loggedUser: UserProfile, issuedApiKey?: string) => Promise<void>;
  handleLogout: () => void;
}

const ConsoleContext = createContext<ConsoleContextValue | null>(null);

export function ConsoleProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname() || "/";
  const { locale, setLocale, t } = useI18n();

  const [authReady, setAuthReady] = useState<boolean>(false);
  const [user, setUser] = useState<UserProfile | null>(null);
  const [activeRole, setActiveRole] = useState<AppRole>("SUPER_ADMIN");
  const [allUsers, setAllUsers] = useState<UserProfile[]>([]);
  const [workers, setWorkers] = useState<GpuWorkerNode[]>([]);
  const [enrollmentTokens, setEnrollmentTokens] = useState<EnrollmentTokenItem[]>([]);
  const [models, setModels] = useState<CatalogModel[]>([]);
  const [rigs, setRigs] = useState<ManagedRig[]>([]);
  const [auditEvents, setAuditEvents] = useState<SecurityAuditEvent[]>([]);
  const [auditCategoryFilter, setAuditCategoryFilter] = useState<string>("ALL");
  const [apiKeys, setApiKeys] = useState<OpenAiApiKeyItem[]>([]);
  const [latestRawApiKey, setLatestRawApiKey] = useState<string>("");
  const [selectedPlaygroundModelId, setSelectedPlaygroundModelId] = useState<string>("");
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const activeSection = useMemo(() => pathToSection(pathname), [pathname]);

  const notify = useCallback((msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 5000);
  }, []);

  const copyText = useCallback((text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  }, []);

  const navigateToSection = useCallback(
    (section: ActiveSection) => {
      const target = SECTION_TO_PATH[section] || "/overview";
      if (pathname !== target) {
        router.push(target);
      }
    },
    [pathname, router]
  );

  const fetchAllData = useCallback(async () => {
    const token = getStoredJwtToken();
    if (!token) {
      setUser(null);
      setAuthReady(true);
      return;
    }
    try {
      const snapshot = await fetchControlPlaneSnapshot();
      if (snapshot.unauthenticated || !snapshot.user) {
        if (snapshot.unauthenticated) {
          setUser(null);
        }
        setAuthReady(true);
        return;
      }
      setUser(snapshot.user);
      const r = (snapshot.user.role as AppRole) || "SUPER_ADMIN";
      setActiveRole(r);
      setAllUsers(snapshot.allUsers || []);
      setWorkers(snapshot.workers || []);
      setEnrollmentTokens(snapshot.tokens || []);
      setModels(snapshot.models || []);
      setRigs(snapshot.rigs || []);
      setApiKeys(snapshot.apiKeys || []);
      setAuditEvents(snapshot.auditEvents || []);
      setAuthReady(true);
    } catch {
      setAuthReady(true);
    }
  }, []);

  useEffect(() => {
    fetchAllData();
    const timer = setInterval(() => {
      if (getStoredJwtToken()) {
        fetchAllData();
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [fetchAllData]);

  // Route guard: redirect unauthenticated users to /login, and authenticated users off / or /login
  useEffect(() => {
    if (!authReady) return;
    if (!user) {
      if (pathname !== "/login") {
        router.replace("/login");
      }
      return;
    }
    if (pathname === "/" || pathname === "/login") {
      router.replace(defaultPathForRole(activeRole));
    }
  }, [authReady, user, activeRole, pathname, router]);

  const handleAuthenticated = useCallback(
    async (loggedUser: UserProfile, issuedApiKey?: string) => {
      setUser(loggedUser);
      if (issuedApiKey) {
        setLatestRawApiKey(issuedApiKey);
      }
      const nextRole = (loggedUser.role as AppRole) || "SUPER_ADMIN";
      setActiveRole(nextRole);
      setAuthReady(true);
      router.replace(defaultPathForRole(nextRole, Boolean(issuedApiKey)));
      notify(
        issuedApiKey
          ? `Accesso effettuato con API Token (${issuedApiKey.slice(0, 16)}...) come ${loggedUser.displayName}`
          : `Accesso effettuato come ${loggedUser.displayName} (${loggedUser.email})`
      );
      await fetchAllData();
    },
    [fetchAllData, notify, router]
  );

  const handleLogout = useCallback(() => {
    clearAuthSession();
    setUser(null);
    setAllUsers([]);
    setEnrollmentTokens([]);
    setApiKeys([]);
    router.replace("/login");
  }, [router]);

  const onlineWorkers = useMemo(
    () => workers.filter((w) => w.assignmentState !== "OFFLINE_STALE"),
    [workers]
  );
  const standbyWorkers = useMemo(
    () => workers.filter((w) => w.assignmentState === "IDLE_STANDBY_POOL"),
    [workers]
  );
  const activeRigs = useMemo(
    () => rigs.filter((r) => r.lifecycleState === "SERVING"),
    [rigs]
  );

  const value: ConsoleContextValue = {
    locale,
    setLocale,
    t,
    authReady,
    user,
    activeRole,
    setActiveRole,
    activeSection,
    navigateToSection,
    allUsers,
    workers,
    onlineWorkers,
    standbyWorkers,
    enrollmentTokens,
    models,
    rigs,
    activeRigs,
    auditEvents,
    auditCategoryFilter,
    setAuditCategoryFilter,
    apiKeys,
    latestRawApiKey,
    setLatestRawApiKey,
    selectedPlaygroundModelId,
    setSelectedPlaygroundModelId,
    toastMessage,
    setToastMessage,
    copiedId,
    notify,
    copyText,
    fetchAllData,
    handleAuthenticated,
    handleLogout
  };

  return <ConsoleContext.Provider value={value}>{children}</ConsoleContext.Provider>;
}

export function useConsole() {
  const ctx = useContext(ConsoleContext);
  if (!ctx) {
    throw new Error("useConsole must be used inside ConsoleProvider");
  }
  return ctx;
}
