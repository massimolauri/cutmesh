"use client";

import React from "react";
import { useConsole } from "@/context/ConsoleContext";
import { LoginView } from "@/components/auth/LoginView";

export default function LoginPage() {
  const { handleAuthenticated } = useConsole();
  return <LoginView onAuthenticated={handleAuthenticated} />;
}
