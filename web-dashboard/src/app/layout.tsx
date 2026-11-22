import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import "katex/dist/katex.min.css";
import { I18nProvider } from "@/i18n/translations";
import { ConsoleProvider } from "@/context/ConsoleContext";
import { ConsoleShell } from "@/components/layout/ConsoleShell";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap"
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap"
});

export const metadata: Metadata = {
  title: "Cutmesh — Distributed AI Cloud",
  description:
    "Cutmesh Enterprise Distributed AI Inference, GPU Cluster Orchestration & OpenAI API Gateway"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="it"
      className={`dark ${inter.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `try{var t=localStorage.getItem("cutmesh_theme");if(t==="light"){document.documentElement.classList.remove("dark");}else{document.documentElement.classList.add("dark");}}catch(e){}`
          }}
        />
      </head>
      <body className="min-h-screen bg-background text-foreground font-sans">
        <I18nProvider>
          <ConsoleProvider>
            <ConsoleShell>{children}</ConsoleShell>
          </ConsoleProvider>
        </I18nProvider>
      </body>
    </html>
  );
}
