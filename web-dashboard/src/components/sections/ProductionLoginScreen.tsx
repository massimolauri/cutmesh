"use client";

import React, { useEffect, useState } from "react";
import { AlertCircle, ArrowRight, Globe, Lock, Mail } from "lucide-react";
import { CutmeshLogo, CutmeshWatermark, CutmeshWordmark } from "@/components/ui/cutmesh-logo";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { UserProfile } from "@/types/control-plane";
import {
  fetchLiveSwarmPublicStatus,
  LiveSwarmPublicStatus,
  loginWithEmailPassword
} from "@/services/control-plane-api";
import { useI18n } from "@/i18n/translations";

export interface ProductionLoginScreenProps {
  onAuthenticated: (user: UserProfile, issuedApiKey?: string) => void;
}

export function ProductionLoginScreen({
  onAuthenticated
}: ProductionLoginScreenProps) {
  const { locale, setLocale } = useI18n();
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [serviceStatus, setServiceStatus] = useState<LiveSwarmPublicStatus>({
    backendOnline: false,
    rttMs: 0,
    onlineWorkersCount: null,
    servingRigsCount: null,
    totalTflops: null
  });

  useEffect(() => {
    let active = true;
    const poll = async () => {
      const status = await fetchLiveSwarmPublicStatus();
      if (active) {
        setServiceStatus(status);
      }
    };
    poll();
    const timer = setInterval(poll, 8000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg(null);
    const res = await loginWithEmailPassword(email, password);
    setLoading(false);
    if (res.user) {
      onAuthenticated(res.user);
    } else {
      setErrorMsg(
        res.error ||
          (locale === "it"
            ? "Credenziali non valide o servizio temporaneamente non raggiungibile."
            : "Invalid credentials or service temporarily unreachable.")
      );
    }
  };

  return (
    <div className="min-h-screen w-full bg-background text-foreground flex flex-col justify-between relative overflow-hidden">
      {/* Visible Swiss geometric watermark */}
      <CutmeshWatermark className="pointer-events-none select-none fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[560px] h-[560px] text-foreground opacity-[0.05]" />

      <header className="relative z-10 w-full px-6 py-4 flex items-center justify-end">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-[11px] font-mono text-muted-foreground px-2.5 py-1 border border-border bg-card rounded-[2px]">
            <span
              className={`h-2 w-2 ${
                serviceStatus.backendOnline ? "bg-emerald-500" : "bg-muted-foreground/50"
              }`}
            />
            <span>{serviceStatus.backendOnline ? "ONLINE" : "OFFLINE"}</span>
          </div>

          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-8 text-xs gap-1.5 font-mono"
            onClick={() => setLocale(locale === "it" ? "en" : "it")}
          >
            <Globe className="w-3.5 h-3.5" />
            {locale.toUpperCase()}
          </Button>
        </div>
      </header>

      <main className="relative z-10 flex-1 flex flex-col items-center justify-center px-4 py-8 gap-6">
        {/* Single Big Hero Brand Lockup */}
        <div className="flex flex-col items-center gap-3.5 text-center">
          <CutmeshLogo size={76} />
          <CutmeshWordmark showIcon={false} className="text-2xl tracking-[0.24em]" />
        </div>

        <Card className="w-full max-w-sm border-border bg-card">
          <CardHeader className="space-y-1.5 pb-4 border-b border-border">
            <CardTitle className="text-base font-bold tracking-tight uppercase">
              {locale === "it" ? "Accedi alla Console" : "Sign in to Console"}
            </CardTitle>
            <CardDescription className="text-xs">
              {locale === "it"
                ? "Console di orchestrazione GPU e inferenza LLM distribuita."
                : "Distributed GPU orchestration and LLM inference console."}
            </CardDescription>
          </CardHeader>

          <form onSubmit={handleSubmit}>
            <CardContent className="space-y-4 pt-5">
              {errorMsg && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription className="text-xs">{errorMsg}</AlertDescription>
                </Alert>
              )}

              <div className="space-y-1.5">
                <Label htmlFor="login-email" className="text-xs">
                  Email
                </Label>
                <div className="relative">
                  <Mail className="w-4 h-4 text-muted-foreground absolute left-3 top-2.5" />
                  <Input
                    id="login-email"
                    type="email"
                    autoComplete="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="name@company.com"
                    className="pl-9 h-9 text-sm"
                    required
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="login-password" className="text-xs">
                  Password
                </Label>
                <div className="relative">
                  <Lock className="w-4 h-4 text-muted-foreground absolute left-3 top-2.5" />
                  <Input
                    id="login-password"
                    type="password"
                    autoComplete="current-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="pl-9 h-9 text-sm"
                    required
                  />
                </div>
              </div>
            </CardContent>

            <CardFooter>
              <Button type="submit" disabled={loading} className="w-full h-9 text-sm gap-2">
                {loading
                  ? locale === "it"
                    ? "Accesso in corso..."
                    : "Signing in..."
                  : locale === "it"
                  ? "Accedi"
                  : "Sign in"}
                {!loading && <ArrowRight className="w-4 h-4" />}
              </Button>
            </CardFooter>
          </form>
        </Card>
      </main>

      <footer className="relative z-10 w-full px-6 py-3.5 text-center text-[11px] font-mono uppercase tracking-widest text-muted-foreground/70">
        DISTRIBUTED GPU INFERENCE
      </footer>
    </div>
  );
}
