import React, { useState, useEffect, useMemo } from 'react';
import { api } from '../api';
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Check, User, Users, ChevronLeft, ChevronRight, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";

const DEFAULT_COUNCIL = [];

// Provider configuration with colors and icons
const PROVIDER_CONFIG = {
    'OpenAI': { color: 'bg-emerald-500', textColor: 'text-emerald-700 dark:text-emerald-400' },
    'Anthropic': { color: 'bg-orange-500', textColor: 'text-orange-700 dark:text-orange-400' },
    'Google': { color: 'bg-blue-500', textColor: 'text-blue-700 dark:text-blue-400' },
    'xAI': { color: 'bg-slate-500', textColor: 'text-slate-700 dark:text-slate-400' },
    'MoonshotAI': { color: 'bg-purple-500', textColor: 'text-purple-700 dark:text-purple-400' },
    'DeepSeek': { color: 'bg-cyan-500', textColor: 'text-cyan-700 dark:text-cyan-400' },
    'Meta': { color: 'bg-indigo-500', textColor: 'text-indigo-700 dark:text-indigo-400' },
    'Mistral': { color: 'bg-rose-500', textColor: 'text-rose-700 dark:text-rose-400' },
    'Other': { color: 'bg-gray-500', textColor: 'text-gray-700 dark:text-gray-400' },
};

// Extract provider from model name (e.g., "OpenAI: GPT-5.2" -> "OpenAI")
function getProvider(modelName) {
    const match = modelName.match(/^([^:]+):/);
    if (match) {
        return match[1].trim();
    }
    // Fallback: try to detect from common patterns
    if (modelName.toLowerCase().includes('gpt') || modelName.toLowerCase().includes('openai')) return 'OpenAI';
    if (modelName.toLowerCase().includes('claude') || modelName.toLowerCase().includes('anthropic')) return 'Anthropic';
    if (modelName.toLowerCase().includes('gemini') || modelName.toLowerCase().includes('google')) return 'Google';
    if (modelName.toLowerCase().includes('grok') || modelName.toLowerCase().includes('xai')) return 'xAI';
    if (modelName.toLowerCase().includes('kimi') || modelName.toLowerCase().includes('moonshot')) return 'MoonshotAI';
    if (modelName.toLowerCase().includes('deepseek')) return 'DeepSeek';
    if (modelName.toLowerCase().includes('llama') || modelName.toLowerCase().includes('meta')) return 'Meta';
    if (modelName.toLowerCase().includes('mistral')) return 'Mistral';
    return 'Other';
}

// Get short model name (remove provider prefix)
function getShortName(modelName) {
    const match = modelName.match(/^[^:]+:\s*(.+)/);
    return match ? match[1].trim() : modelName;
}

export default function ModelSelector({
    isOpen,
    onClose,
    onConfirm,
    initialCouncil = DEFAULT_COUNCIL,
    initialChairman = ''
}) {
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedCouncil, setSelectedCouncil] = useState([]);
    const [selectedChairman, setSelectedChairman] = useState('');
    const [error, setError] = useState(null);
    const [step, setStep] = useState(1);
    const [expandedProviders, setExpandedProviders] = useState({});
    const [activeProvider, setActiveProvider] = useState('all');

    useEffect(() => {
        if (isOpen) {
            fetchModels();
            setSelectedCouncil(initialCouncil.length > 0 ? initialCouncil : []);
            setSelectedChairman(initialChairman);
            setStep(1);
            setActiveProvider('all');
        }
    }, [isOpen, initialCouncil, initialChairman]);

    const fetchModels = async () => {
        try {
            setLoading(true);
            const data = await api.getModels();
            setModels(data.models);

            // Initialize all providers as expanded
            const providers = [...new Set(data.models.map(m => getProvider(m.name)))];
            const expanded = {};
            providers.forEach(p => expanded[p] = true);
            setExpandedProviders(expanded);

            if (initialCouncil.length === 0 && data.models.length > 0) {
                const defaults = data.models
                    .filter(m => m.type === 'council' || m.type === 'both')
                    .slice(0, 5)
                    .map(m => m.id);
                setSelectedCouncil(defaults);
            }

            if (!initialChairman && data.models.length > 0) {
                const defaultChair = data.models.find(m => m.type === 'chairman' || m.type === 'both');
                if (defaultChair) setSelectedChairman(defaultChair.id);
            }

        } catch (err) {
            setError('Failed to load models');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleCouncilToggle = (modelId) => {
        setSelectedCouncil(prev => {
            if (prev.includes(modelId)) {
                return prev.filter(id => id !== modelId);
            } else {
                if (prev.length >= 8) return prev;
                return [...prev, modelId];
            }
        });
    };

    const handleNext = () => {
        if (!selectedChairman) return;
        setStep(2);
        setActiveProvider('all');
    };

    const handleBack = () => {
        setStep(1);
        setActiveProvider('all');
    };

    const handleConfirm = () => {
        if (selectedCouncil.length < 5) return;
        onConfirm(selectedCouncil, selectedChairman);
        onClose();
    };

    const toggleProvider = (provider) => {
        setExpandedProviders(prev => ({
            ...prev,
            [provider]: !prev[provider]
        }));
    };

    const chairmanModels = models.filter(m => m.type === 'chairman' || m.type === 'both');
    const councilModels = models.filter(m => m.type === 'council' || m.type === 'both');

    // Group models by provider
    const groupByProvider = (modelList) => {
        const groups = {};
        modelList.forEach(model => {
            const provider = getProvider(model.name);
            if (!groups[provider]) groups[provider] = [];
            groups[provider].push(model);
        });
        // Sort providers: put providers with more models first
        return Object.entries(groups).sort((a, b) => b[1].length - a[1].length);
    };

    const currentModels = step === 1 ? chairmanModels : councilModels;
    const groupedModels = useMemo(() => groupByProvider(currentModels), [currentModels]);
    const providers = useMemo(() => ['all', ...groupedModels.map(([p]) => p)], [groupedModels]);

    // Filter by active provider
    const filteredGroups = useMemo(() => {
        if (activeProvider === 'all') return groupedModels;
        return groupedModels.filter(([provider]) => provider === activeProvider);
    }, [groupedModels, activeProvider]);

    // Count selected per provider (for council step)
    const getProviderSelectionCount = (provider, models) => {
        return models.filter(m => selectedCouncil.includes(m.id)).length;
    };

    const renderModelCard = (model, isChairman = false) => {
        const isSelected = isChairman
            ? selectedChairman === model.id
            : selectedCouncil.includes(model.id);
        const isDisabled = !isChairman && !isSelected && selectedCouncil.length >= 8;

        return (
            <Card
                key={`${isChairman ? 'chair' : 'council'}-${model.id}`}
                className={cn(
                    "cursor-pointer p-4 transition-all relative",
                    isSelected ? "border-primary bg-primary/5 ring-1 ring-primary" : "hover:border-primary/50",
                    isDisabled && "opacity-50 cursor-not-allowed"
                )}
                onClick={() => {
                    if (isDisabled) return;
                    if (isChairman) {
                        setSelectedChairman(model.id);
                    } else {
                        handleCouncilToggle(model.id);
                    }
                }}
            >
                <div className="flex justify-between items-start mb-2">
                    <div className="font-medium text-sm">{getShortName(model.name)}</div>
                    <div className={cn(
                        "h-5 w-5 rounded flex items-center justify-center transition-colors shrink-0",
                        isChairman ? "rounded-full" : "rounded border",
                        isSelected
                            ? "bg-primary border-primary text-primary-foreground"
                            : "border-muted-foreground"
                    )}>
                        {isSelected && <Check className="h-3 w-3" />}
                    </div>
                </div>
                <div className="text-xs text-muted-foreground space-y-1">
                    <div className="flex gap-3">
                        <span>In: ${model.pricing.input}/M</span>
                        <span>Out: ${model.pricing.output}/M</span>
                    </div>
                    <div className="flex flex-wrap gap-1 mt-2">
                        {model.capabilities.slice(0, 3).map(c => (
                            <span key={c} className="inline-flex items-center rounded-sm border px-1.5 py-0.5 text-[10px] font-medium text-foreground">
                                {c}
                            </span>
                        ))}
                        {model.capabilities.length > 3 && (
                            <span className="text-[10px] text-muted-foreground">+{model.capabilities.length - 3}</span>
                        )}
                    </div>
                </div>
            </Card>
        );
    };

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="max-w-4xl max-h-[90vh] flex flex-col p-0">
                <DialogHeader className="px-6 py-4 border-b">
                    <DialogTitle className="flex items-center gap-3">
                        {step === 1 ? (
                            <>
                                <User className="h-5 w-5 text-primary" />
                                Step 1: Select Chairman
                            </>
                        ) : (
                            <>
                                <Users className="h-5 w-5 text-primary" />
                                Step 2: Select Council Members
                            </>
                        )}
                        <span className="text-sm font-normal text-muted-foreground ml-auto">
                            {step}/2
                        </span>
                    </DialogTitle>
                </DialogHeader>

                {/* Provider Filter Tabs */}
                {!loading && !error && (
                    <div className="px-6 py-2 border-b bg-muted/30">
                        <div className="flex items-center gap-2 overflow-x-auto pb-1">
                            {providers.map(provider => {
                                const config = PROVIDER_CONFIG[provider] || PROVIDER_CONFIG['Other'];
                                const isActive = activeProvider === provider;
                                const count = provider === 'all'
                                    ? currentModels.length
                                    : groupedModels.find(([p]) => p === provider)?.[1]?.length || 0;
                                const selectedCount = provider === 'all'
                                    ? (step === 1 ? (selectedChairman ? 1 : 0) : selectedCouncil.length)
                                    : (step === 1
                                        ? (groupedModels.find(([p]) => p === provider)?.[1]?.some(m => m.id === selectedChairman) ? 1 : 0)
                                        : getProviderSelectionCount(provider, groupedModels.find(([p]) => p === provider)?.[1] || [])
                                    );

                                return (
                                    <Button
                                        key={provider}
                                        variant={isActive ? "default" : "outline"}
                                        size="sm"
                                        className={cn(
                                            "shrink-0 h-8",
                                            isActive && "shadow-sm"
                                        )}
                                        onClick={() => setActiveProvider(provider)}
                                    >
                                        {provider === 'all' ? (
                                            <span>All</span>
                                        ) : (
                                            <>
                                                <span className={cn("w-2 h-2 rounded-full mr-2", config.color)} />
                                                {provider}
                                            </>
                                        )}
                                        <span className="ml-1.5 text-xs opacity-70">
                                            {selectedCount > 0 && step === 2 ? `${selectedCount}/` : ''}{count}
                                        </span>
                                    </Button>
                                );
                            })}
                        </div>
                    </div>
                )}

                <div className="flex-1 min-h-0 overflow-hidden">
                    <ScrollArea className="h-[55vh] px-6 py-4">
                        {loading ? (
                            <div className="flex justify-center items-center h-40">
                                Loading models...
                            </div>
                        ) : error ? (
                            <div className="text-destructive text-center">{error}</div>
                        ) : (
                            <div className="space-y-6">
                                {step === 1 && (
                                    <p className="text-sm text-muted-foreground">
                                        Choose the Chairman who will synthesize the council's responses into a final answer.
                                    </p>
                                )}
                                {step === 2 && (
                                    <div className="flex items-center justify-between">
                                        <p className="text-sm text-muted-foreground">
                                            Select 5-8 models to form your council. They will each provide their perspectives.
                                        </p>
                                        <span className={cn(
                                            "text-sm font-mono px-2 py-1 rounded shrink-0",
                                            selectedCouncil.length >= 5
                                                ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                                                : "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400"
                                        )}>
                                            {selectedCouncil.length}/8 selected
                                        </span>
                                    </div>
                                )}

                                {filteredGroups.map(([provider, providerModels]) => {
                                    const config = PROVIDER_CONFIG[provider] || PROVIDER_CONFIG['Other'];
                                    const isExpanded = expandedProviders[provider] !== false;
                                    const selectedInProvider = step === 1
                                        ? providerModels.some(m => m.id === selectedChairman)
                                        : providerModels.filter(m => selectedCouncil.includes(m.id)).length;

                                    return (
                                        <div key={provider} className="space-y-3">
                                            <button
                                                onClick={() => toggleProvider(provider)}
                                                className="flex items-center gap-2 w-full group"
                                            >
                                                <span className={cn("w-3 h-3 rounded-full shrink-0", config.color)} />
                                                <span className={cn("font-semibold", config.textColor)}>
                                                    {provider}
                                                </span>
                                                <span className="text-xs text-muted-foreground">
                                                    ({providerModels.length} models{selectedInProvider > 0 && `, ${selectedInProvider} selected`})
                                                </span>
                                                <div className="flex-1" />
                                                {isExpanded ? (
                                                    <ChevronUp className="h-4 w-4 text-muted-foreground group-hover:text-foreground transition-colors" />
                                                ) : (
                                                    <ChevronDown className="h-4 w-4 text-muted-foreground group-hover:text-foreground transition-colors" />
                                                )}
                                            </button>

                                            {isExpanded && (
                                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 pl-5">
                                                    {providerModels.map(model => renderModelCard(model, step === 1))}
                                                </div>
                                            )}

                                            <Separator className="mt-4" />
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </ScrollArea>
                </div>

                <DialogFooter className="px-6 py-4 border-t bg-muted/10 flex justify-between">
                    <div>
                        {step === 2 && (
                            <Button variant="ghost" onClick={handleBack}>
                                <ChevronLeft className="h-4 w-4 mr-1" />
                                Back
                            </Button>
                        )}
                    </div>
                    <div className="flex gap-2">
                        <Button variant="outline" onClick={onClose}>Cancel</Button>
                        {step === 1 ? (
                            <Button onClick={handleNext} disabled={!selectedChairman || loading}>
                                Next
                                <ChevronRight className="h-4 w-4 ml-1" />
                            </Button>
                        ) : (
                            <Button onClick={handleConfirm} disabled={selectedCouncil.length < 5 || loading}>
                                Confirm Selection
                            </Button>
                        )}
                    </div>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
