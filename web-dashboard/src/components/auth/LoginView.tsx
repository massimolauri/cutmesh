"use client";

import React from "react";
import { UserProfile } from "@/types/control-plane";
import { ProductionLoginScreen } from "@/components/sections/ProductionLoginScreen";

export interface LoginViewProps {
  onAuthenticated: (user: UserProfile, issuedApiKey?: string) => void;
}

export function LoginView({ onAuthenticated }: LoginViewProps) {
  return <ProductionLoginScreen onAuthenticated={onAuthenticated} />;
}
