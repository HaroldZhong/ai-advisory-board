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
import { Textarea } from '@/components/ui/textarea';
import { Settings, Zap, Brain, Sparkles, ChevronDown, ChevronUp, Shield, ShieldOff, UserCog } from 'lucide-react';
import {
    getAdvancedSettingAvailability,
    isChatOnlyAdvancedOptionDisabled,
    normalizeAdvancedSettingsForMode,
} from '@/utils/advancedSettingsAvailability';
import {
    getResponsiveModalBodyClass,
    getResponsiveModalContentClass,
} from '@/utils/responsiveModalLayout';
import {
    getAdvancedRoutingSummary,
    getModelTierHint,
    getRagPresetHint,
    readAdvancedRoutingDisclosurePreference,
    writeAdvancedRoutingDisclosurePreference,
} from '@/utils/advancedRoutingControls';

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
    nextMessageMode = 'chat',
}) {
    const [localSettings, setLocalSettings] = useState({
        executionMode: settings?.executionMode || 'auto',
        ragPreset: settings?.ragPreset || 'auto',
        modelTier: settings?.modelTier || 'auto',
        zdrEnabled: settings?.zdrEnabled ?? false,
        customInstructions: settings?.customInstructions || '',
    });
    const [isRoutingExpanded, setIsRoutingExpanded] = useState(() => (
        readAdvancedRoutingDisclosurePreference()
    ));

    const availability = getAdvancedSettingAvailability(nextMessageMode);
    const effectiveSettings = normalizeAdvancedSettingsForMode(localSettings, nextMessageMode);
    const hasRoutingOverrides = effectiveSettings.ragPreset !== 'auto' || effectiveSettings.modelTier !== 'auto';

    const setRoutingExpanded = (expanded) => {
        setIsRoutingExpanded(expanded);
        writeAdvancedRoutingDisclosurePreference(expanded);
    };

    const handleSave = () => {
        onSave(effectiveSettings);
        onClose();
    };

    const handleReset = () => {
        setLocalSettings({
            executionMode: 'auto',
            ragPreset: 'auto',
            modelTier: 'auto',
            zdrEnabled: false,
            customInstructions: '',
        });
    };

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className={cn(getResponsiveModalContentClass('form'), "flex flex-col")}>
                <DialogHeader className="shrink-0">
                    <DialogTitle className="flex items-center gap-2">
                        <Settings className="h-5 w-5" />
                        Advanced Settings
                    </DialogTitle>
                </DialogHeader>

                <div className={cn(getResponsiveModalBodyClass(), "space-y-6 py-4 pr-1")}>
                    {/* Execution Mode */}
                    <div>
                        <label className="text-sm font-medium mb-2 block">
                            Execution Mode
                        </label>
                        {availability.notice && (
                            <div className="mb-2 rounded bg-muted/60 border px-2 py-1.5 text-xs text-muted-foreground">
                                {availability.notice}
                            </div>
                        )}
                        <div className="grid grid-cols-2 gap-2">
                            {EXECUTION_MODES.map((mode) => {
                                const Icon = mode.icon;
                                const isSelected = effectiveSettings.executionMode === mode.id;
                                const isDisabled = isChatOnlyAdvancedOptionDisabled('executionMode', mode.id, nextMessageMode);
                                return (
                                    <button
                                        key={mode.id}
                                        disabled={isDisabled}
                                        onClick={() => setLocalSettings(s => ({ ...s, executionMode: mode.id }))}
                                        className={cn(
                                            "flex items-center gap-2 p-3 rounded-lg border text-left transition-all",
                                            isSelected
                                                ? "border-primary bg-primary/5"
                                                : "border-muted hover:border-primary/50",
                                            isDisabled && "opacity-50 cursor-not-allowed hover:border-muted"
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

                    {/* Advanced Routing Controls */}
                    <div className="rounded-lg border border-muted">
                        <button
                            type="button"
                            onClick={() => setRoutingExpanded(!isRoutingExpanded)}
                            aria-expanded={isRoutingExpanded}
                            aria-controls="advanced-routing-controls"
                            className={cn(
                                "w-full flex items-center justify-between gap-3 p-3 text-left transition-colors",
                                hasRoutingOverrides ? "bg-primary/5" : "hover:bg-muted/40"
                            )}
                        >
                            <div className="min-w-0">
                                <div className="text-sm font-medium flex items-center gap-2">
                                    Override auto-routing
                                    {hasRoutingOverrides && (
                                        <span className="text-xs px-1.5 py-0.5 rounded bg-primary/10 text-primary">
                                            Custom
                                        </span>
                                    )}
                                </div>
                                <div className="text-xs text-muted-foreground mt-0.5">
                                    {getAdvancedRoutingSummary(effectiveSettings)}
                                </div>
                            </div>
                            {isRoutingExpanded ? (
                                <ChevronUp className="h-4 w-4 shrink-0 text-muted-foreground" />
                            ) : (
                                <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
                            )}
                        </button>

                        {isRoutingExpanded && (
                            <div id="advanced-routing-controls" className="space-y-5 border-t border-muted p-3">
                                {/* RAG Context Level */}
                                <div>
                                    <label className="text-sm font-medium mb-2 block">
                                        Context Level
                                    </label>
                                    <div className="grid gap-2">
                                        {RAG_PRESETS.map((preset) => {
                                            const isSelected = effectiveSettings.ragPreset === preset.id;
                                            const isDisabled = isChatOnlyAdvancedOptionDisabled('ragPreset', preset.id, nextMessageMode);
                                            return (
                                                <button
                                                    key={preset.id}
                                                    type="button"
                                                    disabled={isDisabled}
                                                    onClick={() => setLocalSettings(s => ({ ...s, ragPreset: preset.id }))}
                                                    className={cn(
                                                        "rounded-lg border p-3 text-left transition-all",
                                                        isSelected
                                                            ? "border-primary bg-primary/5"
                                                            : "border-muted hover:border-primary/50",
                                                        isDisabled && "opacity-50 cursor-not-allowed hover:border-muted"
                                                    )}
                                                >
                                                    <div className="flex items-center justify-between gap-2">
                                                        <span className="text-sm font-medium">{preset.label}</span>
                                                        <span className="text-xs text-muted-foreground">{preset.tokens}</span>
                                                    </div>
                                                    <div className="text-xs text-muted-foreground mt-1">
                                                        {getRagPresetHint(preset.id)}
                                                    </div>
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
                                    <div className="grid gap-2">
                                        {MODEL_TIERS.map((tier) => {
                                            const isSelected = localSettings.modelTier === tier.id;
                                            return (
                                                <button
                                                    key={tier.id}
                                                    type="button"
                                                    onClick={() => setLocalSettings(s => ({ ...s, modelTier: tier.id }))}
                                                    className={cn(
                                                        "rounded-lg border p-3 text-left transition-all",
                                                        isSelected
                                                            ? "border-primary bg-primary/5"
                                                            : "border-muted hover:border-primary/50"
                                                    )}
                                                >
                                                    <div className="text-sm font-medium">{tier.label}</div>
                                                    <div className="text-xs text-muted-foreground mt-1">
                                                        {getModelTierHint(tier.id)}
                                                    </div>
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Privacy - ZDR default */}
                    <div>
                        <label className="text-sm font-medium mb-2 block">
                            Privacy Default
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
                                    Default ZDR for new conversations
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
                                    Existing conversations use their own privacy setting when present
                                </div>
                            </div>
                        </button>

                        {/* ZDR Info Note */}
                        {!localSettings.zdrEnabled && (
                            <div className="mt-2 p-2 rounded bg-amber-500/10 border border-amber-500/20 text-xs text-amber-700 dark:text-amber-400">
                                <strong>Default off:</strong> New standard conversations may use providers according
                                to their retention policies. Use the trust row to change privacy for the active conversation.
                            </div>
                        )}
                        {localSettings.zdrEnabled && (
                            <div className="mt-2 p-2 rounded bg-green-500/10 border border-green-500/20 text-xs text-green-700 dark:text-green-400">
                                <strong>Default on:</strong> New conversations and legacy conversations without
                                stored privacy metadata will prefer Zero Data Retention routes.
                            </div>
                        )}
                    </div>

                    {/* Custom Personas / Instructions */}
                    <div>
                        <label className="text-sm font-medium mb-2 flex items-center gap-2">
                            <UserCog className="h-4 w-4 text-muted-foreground" />
                            Custom Instructions
                        </label>
                        <Textarea
                            value={localSettings.customInstructions}
                            onChange={(e) => setLocalSettings(s => ({ ...s, customInstructions: e.target.value }))}
                            placeholder="Example: Always respond in bullet points. Focus on practical advice. Use a formal tone."
                            className="min-h-[80px] max-h-[200px] resize-y text-sm"
                            rows={3}
                        />
                        <p className="text-xs text-muted-foreground mt-1">
                            These instructions are prepended to every council prompt as a system persona.
                        </p>
                    </div>

                    {/* Info box */}
                    <div className="p-3 rounded-lg bg-muted/50 text-sm text-muted-foreground">
                        <strong>Note:</strong> Auto routing uses query type and session budget.
                        Manual routing overrides can increase context, model quality, and cost.
                    </div>
                </div>

                <DialogFooter className="flex shrink-0 justify-between">
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
