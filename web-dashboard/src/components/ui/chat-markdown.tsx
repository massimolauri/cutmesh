"use client";

import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Check, Copy, ExternalLink, Terminal } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

interface ChatMarkdownProps {
  content: string;
  isStreaming?: boolean;
  className?: string;
}

/**
 * Normalizes LaTeX delimiters produced by Qwen / DeepSeek / OpenAI models
 * (`\[ ... \]` -> `$$ ... $$` and `\( ... \)` -> `$ ... $`) outside of code blocks
 * so `remark-math` + `rehype-katex` render all formulas seamlessly.
 */
function normalizeMathDelimiters(raw: string): string {
  if (!raw) return "";
  const segments = raw.split(/(```[\s\S]*?```|`[^`\n]+`)/g);
  return segments
    .map((seg) => {
      if (seg.startsWith("```") || seg.startsWith("`")) {
        return seg;
      }
      return seg
        .replace(/\\\[([\s\S]*?)\\\]/g, (_m, body) => `\n$$\n${body.trim()}\n$$\n`)
        .replace(/\\\(([\s\S]*?)\\\)/g, (_m, body) => `$${body.trim()}$`);
    })
    .join("");
}

export function CodeBlock({
  language,
  value,
  className
}: {
  language: string;
  value: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  const lang = (language || "text").toLowerCase();

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // ignore clipboard errors
    }
  };

  return (
    <Card
      className={cn(
        "my-2.5 overflow-hidden border-zinc-800 bg-zinc-950 text-zinc-100 shadow-sm",
        className
      )}
    >
      <CardHeader className="flex flex-row items-center justify-between space-y-0 px-3 py-1.5 bg-zinc-900/90 border-b border-zinc-800">
        <div className="flex items-center gap-1.5">
          <Terminal className="h-3.5 w-3.5 text-emerald-400" />
          <Badge
            variant="outline"
            className="h-5 px-1.5 py-0 text-[10px] font-mono uppercase tracking-wider border-zinc-700 bg-zinc-950/60 text-zinc-200"
          >
            {lang}
          </Badge>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={handleCopy}
          className="h-6 px-2 text-[10px] gap-1 text-zinc-300 hover:text-white hover:bg-zinc-800"
        >
          {copied ? (
            <>
              <Check className="h-3 w-3 text-emerald-400" />
              <span className="text-emerald-400">Copiato</span>
            </>
          ) : (
            <>
              <Copy className="h-3 w-3" />
              <span>Copia codice</span>
            </>
          )}
        </Button>
      </CardHeader>
      <CardContent className="p-0 overflow-x-auto">
        <SyntaxHighlighter
          language={lang}
          style={oneDark}
          customStyle={{
            margin: 0,
            padding: "0.75rem",
            background: "transparent",
            fontSize: "0.75rem",
            lineHeight: "1.5"
          }}
          wrapLongLines={false}
        >
          {value}
        </SyntaxHighlighter>
      </CardContent>
    </Card>
  );
}

export function ChatMarkdown({
  content,
  isStreaming = false,
  className = ""
}: ChatMarkdownProps) {
  if (!content) {
    return isStreaming ? (
      <Badge variant="outline" className="font-mono text-[10px] animate-pulse">
        ▍
      </Badge>
    ) : null;
  }

  const normalized = normalizeMathDelimiters(content);

  return (
    <div
      className={cn(
        "text-xs sm:text-[13px] leading-relaxed break-words space-y-2",
        className
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
        components={{
          h1: ({ children }: { children?: React.ReactNode }) => (
            <div className="mt-3 mb-1.5 space-y-1">
              <h1 className="text-base font-bold tracking-tight text-foreground">
                {children}
              </h1>
              <Separator />
            </div>
          ),
          h2: ({ children }: { children?: React.ReactNode }) => (
            <div className="mt-3 mb-1.5 space-y-1">
              <h2 className="text-sm font-bold tracking-tight text-foreground">
                {children}
              </h2>
              <Separator className="bg-border/60" />
            </div>
          ),
          h3: ({ children }: { children?: React.ReactNode }) => (
            <h3 className="text-[13px] font-bold mt-2.5 mb-1 text-foreground">
              {children}
            </h3>
          ),
          h4: ({ children }: { children?: React.ReactNode }) => (
            <h4 className="text-xs font-semibold mt-2 mb-0.5 text-foreground">
              {children}
            </h4>
          ),
          p: ({ children }: { children?: React.ReactNode }) => (
            <p className="leading-relaxed my-1.5">{children}</p>
          ),
          ul: ({ children }: { children?: React.ReactNode }) => (
            <ul className="my-1.5 list-disc pl-5 space-y-1">{children}</ul>
          ),
          ol: ({ children }: { children?: React.ReactNode }) => (
            <ol className="my-1.5 list-decimal pl-5 space-y-1">{children}</ol>
          ),
          li: ({ children }: { children?: React.ReactNode }) => (
            <li className="leading-relaxed">{children}</li>
          ),
          blockquote: ({ children }: { children?: React.ReactNode }) => (
            <Card className="my-2 border-l-4 border-l-primary bg-muted/30 shadow-none rounded-md">
              <CardContent className="p-2.5 pl-3 italic text-muted-foreground text-xs">
                {children}
              </CardContent>
            </Card>
          ),
          hr: () => <Separator className="my-3" />,
          a: ({ href, children }: { href?: string; children?: React.ReactNode }) => (
            <Button
              variant="link"
              size="sm"
              asChild
              className="h-auto p-0 text-xs font-medium inline-flex items-center gap-0.5 align-baseline"
            >
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
                <ExternalLink className="h-2.5 w-2.5 ml-0.5 inline" />
              </a>
            </Button>
          ),
          input: ({ type, checked }: { type?: string; checked?: boolean }) => {
            if (type === "checkbox") {
              return (
                <Checkbox
                  checked={Boolean(checked)}
                  disabled
                  className="mr-1.5 align-middle translate-y-[-1px]"
                />
              );
            }
            return null;
          },
          table: ({ children }: { children?: React.ReactNode }) => (
            <Card className="my-2.5 overflow-hidden shadow-none">
              <Table className="text-xs">{children}</Table>
            </Card>
          ),
          thead: ({ children }: { children?: React.ReactNode }) => (
            <TableHeader className="bg-muted/50">{children}</TableHeader>
          ),
          tbody: ({ children }: { children?: React.ReactNode }) => (
            <TableBody>{children}</TableBody>
          ),
          tr: ({ children }: { children?: React.ReactNode }) => (
            <TableRow>{children}</TableRow>
          ),
          th: ({ children }: { children?: React.ReactNode }) => (
            <TableHead className="h-8 px-3 py-1.5 text-xs font-semibold text-foreground">
              {children}
            </TableHead>
          ),
          td: ({ children }: { children?: React.ReactNode }) => (
            <TableCell className="px-3 py-1.5 text-xs">{children}</TableCell>
          ),
          code: ({
            className: codeClass,
            children
          }: {
            className?: string;
            children?: React.ReactNode;
          }) => {
            const match = /language-([^\s]+)/.exec(codeClass || "");
            const rawCode = String(children).replace(/\n$/, "");
            const isBlock = Boolean(match) || rawCode.includes("\n");
            if (isBlock) {
              return <CodeBlock language={match?.[1] || "text"} value={rawCode} />;
            }
            return (
              <Badge
                variant="secondary"
                className="px-1.5 py-0 mx-0.5 rounded font-mono text-[11px] font-normal align-baseline"
              >
                {children}
              </Badge>
            );
          },
          pre: ({ children }: { children?: React.ReactNode }) => <>{children}</>
        }}
      >
        {normalized}
      </ReactMarkdown>
      {isStreaming && (
        <span className="inline-block ml-0.5 text-primary animate-pulse">▍</span>
      )}
    </div>
  );
}
