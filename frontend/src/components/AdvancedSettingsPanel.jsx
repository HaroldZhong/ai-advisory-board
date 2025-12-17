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
import { Settings, Zap, Brain, Sparkles, ChevronDown, ChevronUp, Shield, ShieldOff } from 'lucide-react';

/**
 * Advanced Settings Panel
 * 
 * Provides power users with fine-grained control over:
 * - Execution mode override
 * - RAG context level
 * - Model tier preference
 * 
 * Hidden by default, accessible via "Advanced" toggle.
 */

const EXECUTION_MODES = [
    {
        id: 'auto',
        label: 'Auto',
        description: 'System decides based on query',
        icon: Sparkles,
    },
    {
        id: 'quick',
        label: 'Quick Answer',
        description: 'Fast, concise responses',
        icon: Zap,
    },
    {
        id: 'standard',
        label: 'Work Mode',
        description: 'Balanced quality and cost',
        icon: Brain,
    },
    {
        id: 'research',
        label: 'Research',
        description: 'Thorough with full context',
        icon: Brain,
    },
];

const RAG_PRESETS = [
    { id: 'auto', label: 'Auto', tokens: 'Varies' },
    { id: 'low', label: 'Minimal', tokens: '4k' },
    { id: 'medium', label: 'Balanced', tokens: '8k' },
    { id: 'high', label: 'Extended', tokens: '16k' },
    { id: 'max', label: 'Maximum', tokens: '32k' },
];

const MODEL_TIERS = [
    { id: 'auto', label: 'Auto', description: 'Based on task' },
    { id: 'budget', label: 'Economy', description: 'Fastest, lowest cost' },
    { id: 'mid', label: 'Balanced', description: 'Good quality/cost ratio' },
    { id: 'premium', label: 'Premium', description: 'Highest quality' },
];

export default function AdvancedSettingsPanel({
    isOpen,
    onClose,
    settings,
    onSave,
}) {
    const [localSettings, setLocalSettings] = useState({
        executionMode: settings?.executionMode || 'auto',
        ragPreset: settings?.ragPreset || 'auto',
        modelTier: settings?.modelTier || 'auto',
        zdrEnabled: settings?.zdrEnabled ?? false,
    });

    const handleSave = () => {
        onSave(localSettings);
        onClose();
    };

    const handleReset = () => {
        setLocalSettings({
            executionMode: 'auto',
            ragPreset: 'auto',
            modelTier: 'auto',
            zdrEnabled: false,
        });
    };

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="sm:max-w-lg">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <Settings className="h-5 w-5" />
                        Advanced Settings
                    </DialogTitle>
                </DialogHeader>

                <div className="py-4 space-y-6">
                    {/* Execution Mode */}
                    <div>
                        <label className="text-sm font-medium mb-2 block">
                            Execution Mode
                        </label>
                        <div className="grid grid-cols-2 gap-2">
                            {EXECUTION_MODES.map((mode) => {
                                const Icon = mode.icon;
                                const isSelected = localSettings.executionMode === mode.id;
                                return (
                                    <button
                                        key={mode.id}
                                        onClick={() => setLocalSettings(s => ({ ...s, executionMode: mode.id }))}
                                        className={cn(
                                            "flex items-center gap-2 p-3 rounded-lg border text-left transition-all",
                                            isSelected
                                                ? "border-primary bg-primary/5"
                                                : "border-muted hover:border-primary/50"
                                        )}
                                    >
                                        <Icon className="h-4 w-4 text-muted-foreground" />
                                        <div>
                                            <div className="font-medium text-sm">{mode.label}</div>
                                            <div className="text-xs text-muted-foreground">{mode.description}</div>
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* RAG Context Level */}
                    <div>
                        <label className="text-sm font-medium mb-2 block">
                            Context Level
                        </label>
                        <div className="flex flex-wrap gap-2">
                            {RAG_PRESETS.map((preset) => {
                                const isSelected = localSettings.ragPreset === preset.id;
                                return (
                                    <button
                                        key={preset.id}
                                        onClick={() => setLocalSettings(s => ({ ...s, ragPreset: preset.id }))}
                                        className={cn(
                                            "px-3 py-2 rounded-lg border text-sm transition-all",
                                            isSelected
                                                ? "border-primary bg-primary/5"
                                                : "border-muted hover:border-primary/50"
                                        )}
                                    >
                                        {preset.label}
                                        <span className="text-xs text-muted-foreground ml-1">
                                            ({preset.tokens})
                                        </span>
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Model Tier */}
                    <div>
                        <label className="text-sm font-medium mb-2 block">
                            Model Tier
                        </label>
                        <div className="flex flex-wrap gap-2">
                            {MODEL_TIERS.map((tier) => {
                                const isSelected = localSettings.modelTier === tier.id;
                                return (
                                    <button
                                        key={tier.id}
                                        onClick={() => setLocalSettings(s => ({ ...s, modelTier: tier.id }))}
                                        className={cn(
                                            "px-3 py-2 rounded-lg border text-sm transition-all",
                                            isSelected
                                                ? "border-primary bg-primary/5"
                                                : "border-muted hover:border-primary/50"
                                        )}
                                    >
                                        {tier.label}
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Privacy - ZDR Toggle */}
                    <div>
                        <label className="text-sm font-medium mb-2 block">
                            Privacy Settings
                        </label>
                        <button
                            onClick={() => setLocalSettings(s => ({ ...s, zdrEnabled: !s.zdrEnabled }))}
                            className={cn(
                                "w-full flex items-center gap-3 p-3 rounded-lg border text-left transition-all",
                                localSettings.zdrEnabled
                                    ? "border-green-500 bg-green-500/10"
                                    : "border-muted hover:border-primary/50"
                            )}
                        >
                            {localSettings.zdrEnabled ? (
                                <Shield className="h-5 w-5 text-green-500" />
                            ) : (
                                <ShieldOff className="h-5 w-5 text-muted-foreground" />
                            )}
                            <div className="flex-1">
                                <div className="font-medium text-sm flex items-center gap-2">
                                    Zero Data Retention (ZDR)
                                    <span className={cn(
                                        "text-xs px-1.5 py-0.5 rounded",
                                        localSettings.zdrEnabled
                                            ? "bg-green-500/20 text-green-600"
                                            : "bg-muted text-muted-foreground"
                                    )}>
                                        {localSettings.zdrEnabled ? 'ON' : 'OFF'}
                                    </span>
                                </div>
                                <div className="text-xs text-muted-foreground">
                                    Prevents providers from storing your data
                                </div>
                            </div>
                        </button>

                        {/* ZDR Info Note */}
                        {!localSettings.zdrEnabled && (
                            <div className="mt-2 p-2 rounded bg-amber-500/10 border border-amber-500/20 text-xs text-amber-700 dark:text-amber-400">
                                <strong>Note:</strong> With ZDR off, providers may store your data according to their retention policies.
                                Enable ZDR for enhanced privacy, but note that some providers may be unavailable.
                            </div>
                        )}
                        {localSettings.zdrEnabled && (
                            <div className="mt-2 p-2 rounded bg-green-500/10 border border-green-500/20 text-xs text-green-700 dark:text-green-400">
                                <strong>Privacy protected:</strong> Your data will only be routed to providers with Zero Data Retention policies.
                            </div>
                        )}
                    </div>

                    {/* Info box */}
                    <div className="p-3 rounded-lg bg-muted/50 text-sm text-muted-foreground">
                        <strong>Note:</strong> These settings override the automatic budget-aware
                        routing. Use "Auto" for each to let the system optimize based on your
                        session budget and query type.
                    </div>
                </div>

                <DialogFooter className="flex justify-between">
                    <Button variant="ghost" onClick={handleReset}>
                        Reset to Auto
                    </Button>
                    <div className="flex gap-2">
                        <Button variant="outline" onClick={onClose}>
                            Cancel
                        </Button>
                        <Button onClick={handleSave}>
                            Save Settings
                        </Button>
                    </div>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}

// Compact toggle for showing advanced settings
export function AdvancedSettingsToggle({ onClick, hasOverrides }) {
    return (
        <button
            onClick={onClick}
            className={cn(
                "flex items-center gap-1 text-xs transition-colors",
                hasOverrides
                    ? "text-primary"
                    : "text-muted-foreground hover:text-foreground"
            )}
        >
            <Settings className="h-3 w-3" />
            Advanced
            {hasOverrides && (
                <span className="w-1.5 h-1.5 rounded-full bg-primary" />
            )}
        </button>
    );
}
