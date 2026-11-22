"use client";

import React, { useState } from "react";
import { OpenAiApiKeyItem } from "@/types/control-plane";
import { useI18n } from "@/i18n/translations";
import {
  API_BASE,
  createOpenAiApiKey,
  revokeOpenAiApiKey
} from "@/services/control-plane-api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CodeBlock } from "@/components/ui/chat-markdown";
import { Input } from "@/components/ui/input";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Code2, Copy, KeyRound, Trash2 } from "lucide-react";

interface ApiKeysSdkSectionProps {
  t: Record<string, string>;
  apiKeys: OpenAiApiKeyItem[];
  latestRawApiKey: string;
  onSetLatestRawApiKey: (rawKey: string) => void;
  copiedId: string | null;
  onCopyText: (text: string, id: string) => void;
  onNotify: (msg: string) => void;
  onRefresh: () => void;
}

export function ApiKeysSdkSection({
  t,
  apiKeys,
  latestRawApiKey,
  onSetLatestRawApiKey,
  copiedId,
  onCopyText,
  onNotify,
  onRefresh
}: ApiKeysSdkSectionProps) {
  const [newKeyName, setNewKeyName] = useState<string>("");
  const [newKeyRpm, setNewKeyRpm] = useState<number>(120);
  const [showRevokedKeys, setShowRevokedKeys] = useState<boolean>(false);

  const revokedKeysCount = apiKeys.filter((k) => k.revoked).length;
  const visibleApiKeys = showRevokedKeys
    ? apiKeys
    : apiKeys.filter((k) => !k.revoked);

  const sampleKeyForSnippet = latestRawApiKey || "<TUA_API_KEY>";

  const handleCreateKey = async () => {
    const label = newKeyName.trim() || "api-key";
    const created = await createOpenAiApiKey(label, newKeyRpm);
    if (created) {
      setNewKeyName("");
      onSetLatestRawApiKey(created.rawApiKeySecret);
      onNotify(`API Key '${created.keyPrefix}' creata`);
      onRefresh();
    }
  };

  const { locale } = useI18n();
  const isEn = locale === "en";

  const handleRevokeKey = async (keyId: string) => {
    await revokeOpenAiApiKey(keyId);
    onNotify(isEn ? "API key revoked" : "Chiave API revocata");
    onRefresh();
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-bold">{t.apiKeysTitle}</h2>
        <p className="text-xs text-muted-foreground">{t.apiKeysSubtitle}</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">
            {isEn ? "Generate New API Key" : "Genera Nuova Chiave API"}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Input
              value={newKeyName}
              onChange={(e) => setNewKeyName(e.target.value)}
              placeholder={isEn ? "API key name" : "Nome chiave API"}
            />
            <Select
              value={String(newKeyRpm)}
              onValueChange={(v) => setNewKeyRpm(Number(v))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="60">Rate Limit: 60 RPM</SelectItem>
                <SelectItem value="120">Rate Limit: 120 RPM</SelectItem>
                <SelectItem value="600">Rate Limit: 600 RPM (Enterprise)</SelectItem>
              </SelectContent>
            </Select>
            <Button onClick={handleCreateKey} className="gap-2">
              <KeyRound className="h-4 w-4" />
              {isEn ? "Generate API Key" : "Genera Chiave API"}
            </Button>
          </div>

          {latestRawApiKey && (
            <Alert>
              <KeyRound className="h-4 w-4" />
              <AlertTitle>
                {isEn
                  ? "API Key Generated (Copy it now: it won't be shown again)"
                  : "Chiave API Generata (Copiala ora: non verrà mostrata nuovamente)"}
              </AlertTitle>
              <AlertDescription className="flex items-center gap-2 mt-2">
                <Input
                  readOnly
                  value={latestRawApiKey}
                  className="flex-1 h-8 text-xs font-mono bg-muted"
                />
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => onCopyText(latestRawApiKey, "raw-key")}
                >
                  <Copy className="h-3.5 w-3.5 mr-1" />
                  {copiedId === "raw-key"
                    ? isEn
                      ? "Copied!"
                      : "Copiata!"
                    : isEn
                    ? "Copy Key"
                    : "Copia Chiave"}
                </Button>
              </AlertDescription>
            </Alert>
          )}

          <div className="flex items-center justify-between pt-2">
            <span className="text-xs font-semibold text-muted-foreground">
              {isEn
                ? `Active keys (${apiKeys.filter((k) => !k.revoked).length})`
                : `Chiavi attive (${apiKeys.filter((k) => !k.revoked).length})`}
            </span>
            {revokedKeysCount > 0 && (
              <Button
                variant="outline"
                size="sm"
                className="h-7 text-[11px] font-mono"
                onClick={() => setShowRevokedKeys((v) => !v)}
              >
                {showRevokedKeys
                  ? isEn
                    ? `Hide revoked (${revokedKeysCount})`
                    : `Nascondi revocate (${revokedKeysCount})`
                  : isEn
                  ? `Show revoked (${revokedKeysCount})`
                  : `Mostra revocate (${revokedKeysCount})`}
              </Button>
            )}
          </div>

          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{isEn ? "Key Name" : "Nome Chiave"}</TableHead>
                <TableHead>{isEn ? "Prefix" : "Prefisso"}</TableHead>
                <TableHead>Rate Limit</TableHead>
                <TableHead>{isEn ? "Requests Served" : "Richieste Servite"}</TableHead>
                <TableHead>{isEn ? "Tokens Generated" : "Token Generati"}</TableHead>
                <TableHead>{isEn ? "Status" : "Stato"}</TableHead>
                <TableHead></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {visibleApiKeys.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-6 text-xs text-muted-foreground">
                    {isEn ? "No active API keys." : "Nessuna chiave API attiva."}
                  </TableCell>
                </TableRow>
              ) : (
                visibleApiKeys.map((k) => (
                  <TableRow key={k.id} className={k.revoked ? "opacity-60" : ""}>
                    <TableCell className="font-semibold">{k.name}</TableCell>
                    <TableCell className="font-mono">{k.keyPrefix}</TableCell>
                    <TableCell>{k.rateLimitRpm} RPM</TableCell>
                    <TableCell className="font-mono">{k.totalRequests}</TableCell>
                    <TableCell className="font-mono">{k.totalTokensGenerated}</TableCell>
                    <TableCell>
                      <Badge variant={k.revoked ? "destructive" : "default"}>
                        {k.revoked ? "REVOKED" : "ACTIVE"}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      {!k.revoked && (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleRevokeKey(k.id)}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* SDK Integration Snippets */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Code2 className="h-4 w-4" />
            {isEn ? "OpenAI Standard SDK Quickstart" : "Quickstart Integrazione SDK OpenAI Standard"}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="python">
            <TabsList>
              <TabsTrigger value="python">Python (openai SDK)</TabsTrigger>
              <TabsTrigger value="typescript">TypeScript / Node.js</TabsTrigger>
              <TabsTrigger value="curl">cURL</TabsTrigger>
            </TabsList>
            <TabsContent value="python">
              <CodeBlock
                language="python"
                value={`from openai import OpenAI

client = OpenAI(
    base_url="${API_BASE}/v1",
    api_key="${sampleKeyForSnippet}"
)

response = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Riassumi i vantaggi dell'inferenza distribuita."}],
    extra_body={"conversation_id": "sticky-session-01"}
)
print(response.choices[0].message.content)`}
              />
            </TabsContent>
            <TabsContent value="typescript">
              <CodeBlock
                language="typescript"
                value={`import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "${API_BASE}/v1",
  apiKey: "${sampleKeyForSnippet}",
});

const res = await client.chat.completions.create({
  model: "qwen3.8-27b",
  messages: [{ role: "user", content: "Hello Cutmesh!" }],
});
console.log(res.choices[0].message.content);`}
              />
            </TabsContent>
            <TabsContent value="curl">
              <CodeBlock
                language="bash"
                value={`curl -X POST "${API_BASE}/v1/chat/completions" \\
  -H "Authorization: Bearer ${sampleKeyForSnippet}" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"qwen3.8-27b","conversation_id":"conv-01","messages":[{"role":"user","content":"Test API"}]}'`}
              />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}

