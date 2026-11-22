"use client"

import * as React from "react"
import { Slot } from "@radix-ui/react-slot"

import { cn } from "@/lib/utils"

interface CollapsibleContextValue {
  open: boolean
  onOpenChange: (open: boolean) => void
  disabled?: boolean
}

const CollapsibleContext = React.createContext<CollapsibleContextValue>({
  open: false,
  onOpenChange: () => {},
  disabled: false,
})

export interface CollapsibleProps extends React.HTMLAttributes<HTMLDivElement> {
  open?: boolean
  defaultOpen?: boolean
  onOpenChange?: (open: boolean) => void
  disabled?: boolean
}

const Collapsible = React.forwardRef<HTMLDivElement, CollapsibleProps>(
  (
    {
      open: controlledOpen,
      defaultOpen = false,
      onOpenChange,
      disabled = false,
      className,
      children,
      ...props
    },
    ref
  ) => {
    const [uncontrolledOpen, setUncontrolledOpen] =
      React.useState<boolean>(defaultOpen)
    const isOpen =
      controlledOpen !== undefined ? controlledOpen : uncontrolledOpen

    const handleOpenChange = React.useCallback(
      (next: boolean) => {
        if (disabled) return
        if (controlledOpen === undefined) {
          setUncontrolledOpen(next)
        }
        onOpenChange?.(next)
      },
      [controlledOpen, disabled, onOpenChange]
    )

    return (
      <CollapsibleContext.Provider
        value={{ open: isOpen, onOpenChange: handleOpenChange, disabled }}
      >
        <div
          ref={ref}
          data-state={isOpen ? "open" : "closed"}
          data-disabled={disabled ? "" : undefined}
          className={cn(className)}
          {...props}
        >
          {children}
        </div>
      </CollapsibleContext.Provider>
    )
  }
)
Collapsible.displayName = "Collapsible"

export interface CollapsibleTriggerProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  asChild?: boolean
}

const CollapsibleTrigger = React.forwardRef<
  HTMLButtonElement,
  CollapsibleTriggerProps
>(({ asChild = false, onClick, className, ...props }, ref) => {
  const { open, onOpenChange, disabled } = React.useContext(CollapsibleContext)
  const Comp = asChild ? Slot : "button"

  return (
    <Comp
      ref={ref}
      type={asChild ? undefined : "button"}
      aria-expanded={open}
      data-state={open ? "open" : "closed"}
      data-disabled={disabled ? "" : undefined}
      disabled={disabled}
      onClick={(e: React.MouseEvent<HTMLButtonElement>) => {
        onOpenChange(!open)
        onClick?.(e)
      }}
      className={cn(className)}
      {...props}
    />
  )
})
CollapsibleTrigger.displayName = "CollapsibleTrigger"

export interface CollapsibleContentProps
  extends React.HTMLAttributes<HTMLDivElement> {
  forceMount?: boolean
}

const CollapsibleContent = React.forwardRef<
  HTMLDivElement,
  CollapsibleContentProps
>(({ forceMount, className, children, ...props }, ref) => {
  const { open } = React.useContext(CollapsibleContext)
  if (!open && !forceMount) return null

  return (
    <div
      ref={ref}
      data-state={open ? "open" : "closed"}
      className={cn(
        "overflow-hidden data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
})
CollapsibleContent.displayName = "CollapsibleContent"

export { Collapsible, CollapsibleTrigger, CollapsibleContent }

