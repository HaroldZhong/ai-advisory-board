import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogFooter,
} from '@/components/ui/dialog';
import { cn } from '@/lib/utils';
import { DollarSign, Zap, Sparkles } from 'lucide-react';

/**
 * Session Budget Selector
 * 
 * Allows users to set a spending limit for their conversation session.
 * Budget presets are designed to be outcome-focused rather than token-focused.
 */

const BUDGET_PRESETS = [
    {
        id: 'off',
        label: 'No Limit',
        description: 'Unlimited spending',
        value: null,
        icon: Sparkles,
        color: 'text-purple-500',
    },
    {
        id: 'light',
        label: '$1',
        description: '~10-15 messages',
        value: 1.00,
        icon: Zap,
        color: 'text-green-500',
    },
    {
        id: 'standard',
        label: '$2',
        description: '~20-30 messages',
        value: 2.00,
        icon: DollarSign,
        color: 'text-blue-500',
        default: true,
    },
    {
        id: 'research',
        label: '$5',
        description: 'Extended research',
        value: 5.00,
        icon: DollarSign,
        color: 'text-amber-500',
    },
];

export default function SessionBudgetSelector({
    isOpen,
    onClose,
    onConfirm,
    currentBudget = null,
}) {
    const [selectedBudget, setSelectedBudget] = useState(
        currentBudget ?? BUDGET_PRESETS.find(p => p.default)?.value ?? null
    );

    const handleConfirm = () => {
        onConfirm(selectedBudget);
        onClose();
    };

    const getSelectedPreset = () => {
        return BUDGET_PRESETS.find(p => p.value === selectedBudget) || BUDGET_PRESETS[0];
    };

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <DollarSign className="h-5 w-5 text-primary" />
                        Session Budget
                    </DialogTitle>
                </DialogHeader>

                <div className="py-4">
                    <p className="text-sm text-muted-foreground mb-4">
                        Set a spending limit for this conversation. You'll receive
                        alerts at 70%, 85%, and 100% of your budget.
                    </p>

                    <div className="grid grid-cols-2 gap-3">
                        {BUDGET_PRESETS.map((preset) => {
                            const Icon = preset.icon;
                            const isSelected = selectedBudget === preset.value;

                            return (
                                <button
                                    key={preset.id}
                                    onClick={() => setSelectedBudget(preset.value)}
                                    className={cn(
                                        "relative flex flex-col items-center p-4 rounded-lg border-2 transition-all",
                                        isSelected
                                            ? "border-primary bg-primary/5"
                                            : "border-muted hover:border-primary/50"
                                    )}
                                >
                                    {preset.default && (
                                        <span className="absolute -top-2 -right-2 text-[10px] bg-primary text-primary-foreground px-1.5 py-0.5 rounded-full">
                                            Recommended
                                        </span>
                                    )}
                                    <Icon className={cn("h-6 w-6 mb-2", preset.color)} />
                                    <span className="font-semibold">{preset.label}</span>
                                    <span className="text-xs text-muted-foreground mt-1">
                                        {preset.description}
                                    </span>
                                </button>
                            );
                        })}
                    </div>

                    <div className="mt-4 p-3 rounded-lg bg-muted/50 text-sm">
                        <p className="text-muted-foreground">
                            <strong>How it works:</strong> When you approach your budget,
                            the system automatically reduces context size and suggests
                            more economical options. Your conversation will never be
                            interrupted—responses continue in a lower-cost mode.
                        </p>
                    </div>
                </div>

                <DialogFooter>
                    <Button variant="outline" onClick={onClose}>
                        Cancel
                    </Button>
                    <Button onClick={handleConfirm}>
                        Set Budget
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}

// Compact inline budget indicator for chat interface
export function BudgetIndicator({ budgetUsd, spentUsd, spentPct, className }) {
    if (budgetUsd === null) {
        return null;  // No budget set
    }

    const getStatusColor = () => {
        if (spentPct === null) return 'bg-green-500';
        if (spentPct >= 1.0) return 'bg-red-500';
        if (spentPct >= 0.85) return 'bg-orange-500';
        if (spentPct >= 0.70) return 'bg-yellow-500';
        return 'bg-green-500';
    };

    const pctDisplay = spentPct !== null ? Math.round(spentPct * 100) : 0;

    return (
        <div className={cn("flex items-center gap-2 text-xs", className)}>
            <div className={cn("w-2 h-2 rounded-full", getStatusColor())} />
            <span className="text-muted-foreground">
                ${spentUsd?.toFixed(2) || '0.00'} / ${budgetUsd.toFixed(2)}
            </span>
            <span className="text-muted-foreground">
                ({pctDisplay}%)
            </span>
        </div>
    );
}

// Budget warning banner for inline display
export function BudgetWarningBanner({ threshold, onDismiss }) {
    const getMessage = () => {
        if (threshold >= 1.0) {
            return "You've reached your session budget. Responses will use a lower-cost mode.";
        }
        if (threshold >= 0.85) {
            return "You're at 85% of your session budget.";
        }
        return "You're at 70% of your session budget.";
    };

    const getColor = () => {
        if (threshold >= 1.0) return 'bg-red-500/10 border-red-500/20 text-red-700 dark:text-red-400';
        if (threshold >= 0.85) return 'bg-orange-500/10 border-orange-500/20 text-orange-700 dark:text-orange-400';
        return 'bg-yellow-500/10 border-yellow-500/20 text-yellow-700 dark:text-yellow-400';
    };

    return (
        <div className={cn("p-3 rounded-lg border mb-3", getColor())}>
            <div className="flex items-center justify-between">
                <span className="text-sm">{getMessage()}</span>
                <button
                    onClick={onDismiss}
                    className="text-xs underline opacity-70 hover:opacity-100"
                >
                    Dismiss
                </button>
            </div>
        </div>
    );
}
