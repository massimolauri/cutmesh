"use client";

import React, { useEffect, useState } from "react";
import { Locale } from "@/i18n/translations";
import { AppRole, UserProfile } from "@/types/control-plane";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from "@/components/ui/tooltip";
import { Globe, LogOut, Moon, Sun } from "lucide-react";
import { CutmeshWordmark } from "@/components/ui/cutmesh-logo";

interface TopHeaderProps {
  t: Record<string, string>;
  locale: Locale;
  onLocaleSwitch: (nextLocale: Locale) => void;
  activeRole: AppRole;
  onRoleSwitch: (nextRole: AppRole) => void;
  user: UserProfile | null;
  onLogout: () => void;
  workersCount: number;
  totalFleetTflops: number;
  activeRigsCount: number;
}

export function TopHeader({
  t,
  locale,
  onLocaleSwitch,
  activeRole,
  onRoleSwitch,
  user,
  onLogout,
  workersCount,
  activeRigsCount
}: TopHeaderProps) {
  const [isDark, setIsDark] = useState<boolean>(true);

  useEffect(() => {
    const saved = window.localStorage.getItem("cutmesh_theme");
    if (saved === "light") {
      document.documentElement.classList.remove("dark");
      setIsDark(false);
    } else {
      document.documentElement.classList.add("dark");
      setIsDark(true);
    }
  }, []);

  const toggleTheme = () => {
    const html = document.documentElement;
    if (html.classList.contains("dark")) {
      html.classList.remove("dark");
      window.localStorage.setItem("cutmesh_theme", "light");
      setIsDark(false);
    } else {
      html.classList.add("dark");
      window.localStorage.setItem("cutmesh_theme", "dark");
      setIsDark(true);
    }
  };

  const roleBadgeLabel =
    activeRole === "SUPER_ADMIN"
      ? t.roleAdmin
      : activeRole === "HARDWARE_PROVIDER"
      ? t.roleProvider
      : t.roleDeveloper;

  return (
    <header className="h-14 border-b border-border bg-background sticky top-0 z-40 px-4 flex items-center justify-between gap-2">
      <div className="flex items-center gap-3">
        <CutmeshWordmark className="text-sm" iconSize={32} />
        <div>
          <div className="flex items-center gap-2">
            <Badge
              variant="outline"
              className="text-[10px] font-mono uppercase tracking-wider border-primary/40 text-primary bg-primary/10"
            >
              {roleBadgeLabel}
            </Badge>
          </div>
          <p className="text-[11px] text-muted-foreground hidden md:block">
            {t.brandSubtitle}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2.5">
        <div className="hidden xl:inline-flex items-center gap-2 py-1 px-2.5 text-[11px] font-mono border border-border bg-card rounded-[2px] text-foreground">
          <span className="h-2 w-2 bg-emerald-500" />
          {workersCount} GPU · {activeRigsCount} {t.activeRigsUnit}
        </div>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={toggleTheme}
            >
              {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            {isDark ? t.lightTheme : t.darkTheme}
          </TooltipContent>
        </Tooltip>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm" className="h-8 gap-1.5 text-xs">
              <Globe className="h-3.5 w-3.5" />
              {locale.toUpperCase()}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel className="text-xs">{t.languageLabel}</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => onLocaleSwitch("it")}>
              Italiano (IT)
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onLocaleSwitch("en")}>
              English (EN)
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        {user && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="flex items-center gap-2 px-2 h-9">
                <Avatar className="h-7 w-7">
                  <AvatarImage src={user.avatarUrl} alt={user.displayName} />
                  <AvatarFallback>
                    {user.displayName.slice(0, 2).toUpperCase()}
                  </AvatarFallback>
                </Avatar>
                <div className="text-left hidden sm:block">
                  <div className="text-xs font-medium leading-none">{user.displayName}</div>
                  <div className="text-[10px] text-muted-foreground font-mono">
                    {user.email}
                  </div>
                </div>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-60">
              <DropdownMenuLabel>
                <div className="text-xs font-semibold">{user.displayName}</div>
                <div className="text-[11px] font-mono text-muted-foreground font-normal">
                  {user.email}
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={onLogout}
                className="text-destructive focus:text-destructive"
              >
                <LogOut className="h-3.5 w-3.5 mr-2" />
                {t.signOut}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>
    </header>
  );
}
