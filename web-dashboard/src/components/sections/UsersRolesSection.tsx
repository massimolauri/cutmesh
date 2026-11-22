"use client";

import React, { useState } from "react";
import {
  Trash2,
  UserPlus,
  Users
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";
import { AppRole, UserProfile } from "@/types/control-plane";
import { useI18n } from "@/i18n/translations";

interface UsersRolesSectionProps {
  users: UserProfile[];
  currentUserId?: string;
  onUpdateUserRole: (userId: string, role: AppRole) => Promise<void>;
  onCreateUser: (payload: {
    email: string;
    password: string;
    displayName: string;
    role: AppRole;
  }) => Promise<void>;
  onDeleteUser: (userId: string) => Promise<void>;
}

export function UsersRolesSection({
  users,
  currentUserId,
  onUpdateUserRole,
  onCreateUser,
  onDeleteUser
}: UsersRolesSectionProps) {
  const { locale } = useI18n();
  const isEn = locale === "en";
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<AppRole>("API_USER");
  const [busy, setBusy] = useState(false);

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !email.trim() || !password.trim()) return;
    setBusy(true);
    await onCreateUser({
      displayName: name.trim(),
      email: email.trim(),
      password,
      role
    });
    setName("");
    setEmail("");
    setPassword("");
    setBusy(false);
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-semibold">
              {isEn ? "Administrators" : "Amministratori"}
            </CardDescription>
            <CardTitle className="text-xl">
              {users.filter((u) => u.role === "SUPER_ADMIN").length} {isEn ? "Users" : "Utenti"}
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground">
            {isEn
              ? "Manages model catalog, GPU cluster allocation, users, and audit logs."
              : "Gestisce il catalogo modelli, l'allocazione dei cluster GPU, gli utenti e i log."}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-semibold">
              {isEn ? "Hardware Providers" : "Fornitori Hardware"}
            </CardDescription>
            <CardTitle className="text-xl">
              {users.filter((u) => u.role === "HARDWARE_PROVIDER").length} {isEn ? "Users" : "Utenti"}
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground">
            {isEn
              ? "Issues node enrollment tokens and monitors connected GPU cards."
              : "Genera token di collegamento e monitora le proprie schede GPU nel cluster."}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-semibold">
              {isEn ? "API Users & Developers" : "Utenti API & Sviluppatori"}
            </CardDescription>
            <CardTitle className="text-xl">
              {users.filter((u) => u.role === "API_USER").length} {isEn ? "Users" : "Utenti"}
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground">
            {isEn
              ? "Creates OpenAI-compatible API keys and uses the AI Chat Playground."
              : "Genera chiavi API compatibili OpenAI e utilizza il Playground Chat."}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Create User Form */}
        <Card className="lg:col-span-4">
          <CardHeader>
            <CardTitle className="text-base">{isEn ? "New User" : "Nuovo Utente"}</CardTitle>
            <CardDescription>
              {isEn
                ? "Create an account and assign its access role."
                : "Crea un account e assegna il relativo profilo di accesso."}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleAdd} className="space-y-3.5">
              <div className="space-y-1.5">
                <Label className="text-xs">{isEn ? "Name / Identifier" : "Nome / Identificativo"}</Label>
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder={isEn ? "Full name" : "Nome completo"}
                  className="h-9 text-xs"
                  required
                />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Email</Label>
                <Input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Email"
                  className="h-9 text-xs"
                  required
                />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Password</Label>
                <Input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="h-9 text-xs"
                  required
                />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">{isEn ? "Role" : "Ruolo"}</Label>
                <Select value={role} onValueChange={(v) => setRole(v as AppRole)}>
                  <SelectTrigger className="h-9 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="SUPER_ADMIN">{isEn ? "Administrator" : "Amministratore"}</SelectItem>
                    <SelectItem value="HARDWARE_PROVIDER">{isEn ? "GPU Provider" : "Fornitore GPU"}</SelectItem>
                    <SelectItem value="API_USER">{isEn ? "API User" : "Utente API"}</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button type="submit" disabled={busy} className="w-full h-9 text-xs">
                <UserPlus className="w-3.5 h-3.5 mr-1.5" />
                {isEn ? "Create User" : "Crea Utente"}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Users Table */}
        <Card className="lg:col-span-8">
          <CardHeader>
            <CardTitle className="text-base">{isEn ? "User Directory" : "Directory Utenti"}</CardTitle>
            <CardDescription>
              {isEn
                ? "Manage registered accounts and update access permissions."
                : "Gestisci gli account registrati e modifica i permessi di accesso."}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{isEn ? "User" : "Utente"}</TableHead>
                  <TableHead>{isEn ? "Auth Method" : "Metodo Accesso"}</TableHead>
                  <TableHead>{isEn ? "Role" : "Ruolo"}</TableHead>
                  <TableHead className="text-right">{isEn ? "Actions" : "Azioni"}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {users.map((u) => (
                  <TableRow key={u.id}>
                    <TableCell>
                      <div className="font-medium text-xs flex items-center gap-1.5">
                        {u.displayName}
                        {u.id === currentUserId && (
                          <Badge variant="secondary" className="text-[10px]">
                            {isEn ? "Active Session" : "Sessione Attiva"}
                          </Badge>
                        )}
                      </div>
                      <div className="text-[11px] text-muted-foreground font-mono">{u.email}</div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className="text-[10px] font-mono">
                        Email & Password
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Select
                        value={(u.role as AppRole) || "API_USER"}
                        onValueChange={(nextRole) => onUpdateUserRole(u.id, nextRole as AppRole)}
                      >
                        <SelectTrigger className="h-8 w-[195px] text-xs font-medium">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="SUPER_ADMIN">{isEn ? "Administrator" : "Amministratore"}</SelectItem>
                          <SelectItem value="HARDWARE_PROVIDER">{isEn ? "GPU Provider" : "Fornitore GPU"}</SelectItem>
                          <SelectItem value="API_USER">{isEn ? "API User" : "Utente API"}</SelectItem>
                        </SelectContent>
                      </Select>
                    </TableCell>
                    <TableCell className="text-right">
                      {users.length > 1 && u.id !== currentUserId && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => onDeleteUser(u.id)}
                          className="h-7 text-xs text-destructive hover:text-destructive"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
