import React, { useEffect, useMemo, useState } from 'react';
import { Check, Lock, SlidersHorizontal, Sparkles, Users } from 'lucide-react';

import { api } from '../api';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { useSettings } from '@/contexts/SettingsContext';
import { cn } from '@/lib/utils';
import {
  estimateSelectionCost,
  filterModelsForRole,
  canConfirmModelSelection,
  canStartPresetWithZdr,
  getEffectivePresetZdr,
  getProvider,
  getShortName,
  groupModelsByProvider,
  modelById,
  resolvePresetModels,
  resolveInitialZdrPreference,
} from '../utils/modelPresets';

const DEFAULT_COUNCIL = [];
const MAX_COUNCIL_SIZE = 8;
const MIN_COUNCIL_SIZE = 3;

const PROVIDER_COLORS = {
  OpenAI: 'bg-emerald-500',
  Anthropic: 'bg-orange-500',
  Google: 'bg-blue-500',
  xAI: 'bg-slate-500',
  MoonshotAI: 'bg-purple-500',
  DeepSeek: 'bg-cyan-500',
  Mistral: 'bg-rose-500',
  MiniMax: 'bg-pink-500',
  Qwen: 'bg-violet-500',
  'Z.ai': 'bg-lime-500',
  Other: 'bg-gray-500',
};

function formatCost(cost) {
  if (cost == null || Number.isNaN(cost)) return 'n/a';
  if (cost < 0.01) return `<$0.01`;
  return `$${cost.toFixed(2)}`;
}

function ModelChip({ model }) {
  if (!model) return null;
  const provider = getProvider(model.name);
  return (
    <span className="inline-flex min-w-0 items-center gap-1 rounded border px-2 py-1 text-xs">
      <span className={cn('h-2 w-2 shrink-0 rounded-full', PROVIDER_COLORS[provider] || PROVIDER_COLORS.Other)} />
      <span className="truncate">{getShortName(model.name)}</span>
    </span>
  );
}

function StatPill({ children, tone = 'default' }) {
  return (
    <span className={cn(
      'inline-flex items-center rounded border px-2 py-1 text-xs font-medium',
      tone === 'green' && 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
      tone === 'amber' && 'border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300',
      tone === 'default' && 'text-muted-foreground',
    )}>
      {children}
    </span>
  );
}

export default function ModelSelector({
  isOpen,
  onClose,
  onConfirm,
  initialCouncil = DEFAULT_COUNCIL,
  initialChairman = '',
  defaultBudgetUsd = null,
}) {
  const { settings } = useSettings();
  const [models, setModels] = useState([]);
  const [presets, setPresets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('presets');
  const [selectedPresetId, setSelectedPresetId] = useState('balanced');
  const [selectedCouncil, setSelectedCouncil] = useState([]);
  const [selectedChairman, setSelectedChairman] = useState('');
  const [zdrEnabled, setZdrEnabled] = useState(false);
  const [customRole, setCustomRole] = useState('council');
  const [activeProvider, setActiveProvider] = useState('all');

  useEffect(() => {
    if (!isOpen) return;

    setLoading(true);
    setError(null);
    setActiveTab('presets');
    setCustomRole('council');
    setActiveProvider('all');
    setZdrEnabled(resolveInitialZdrPreference(settings));

    api.getModels()
      .then((data) => {
        const loadedModels = data.models || [];
        const loadedPresets = data.presets || [];
        setModels(loadedModels);
        setPresets(loadedPresets);

        const defaultPreset = loadedPresets.find((preset) => preset.id === 'balanced') || loadedPresets[0];
        const initialPresetId = defaultPreset?.id || '';
        setSelectedPresetId(initialPresetId);

        const presetSelection = resolvePresetModels(defaultPreset, loadedModels, false);
        setSelectedChairman(initialChairman || presetSelection.chairman?.id || data.defaults?.chairman || '');
        setSelectedCouncil(
          initialCouncil.length > 0
            ? initialCouncil
            : presetSelection.council.map((model) => model.id),
        );
      })
      .catch((err) => {
        setError('Failed to load models');
        console.error(err);
      })
      .finally(() => setLoading(false));
  }, [isOpen, initialChairman, initialCouncil, settings.defaultZdrEnabled, settings.zdrEnabled]);

  const byId = useMemo(() => modelById(models), [models]);
  const selectedPreset = useMemo(
    () => presets.find((preset) => preset.id === selectedPresetId) || null,
    [presets, selectedPresetId],
  );
  const effectivePresetZdr = activeTab === 'presets'
    ? getEffectivePresetZdr(selectedPreset, zdrEnabled)
    : zdrEnabled;
  const presetModels = useMemo(
    () => resolvePresetModels(selectedPreset, models, effectivePresetZdr),
    [effectivePresetZdr, models, selectedPreset],
  );
  const customChairman = byId.get(selectedChairman) || null;
  const customCouncil = selectedCouncil.map((id) => byId.get(id)).filter(Boolean);
  const activeChairman = activeTab === 'presets' ? presetModels.chairman : customChairman;
  const activeCouncil = activeTab === 'presets' ? presetModels.council : customCouncil;
  const selectedPresetAvailable = activeTab !== 'presets'
    || canStartPresetWithZdr(selectedPreset, models, effectivePresetZdr);
  const zdrToggleLocked = activeTab === 'presets' && selectedPreset?.requires_zdr;
  const estimatedCost = estimateSelectionCost({ chairman: activeChairman, council: activeCouncil });
  const roleModels = useMemo(
    () => filterModelsForRole(models, customRole, zdrEnabled),
    [models, customRole, zdrEnabled],
  );
  const groupedModels = useMemo(() => groupModelsByProvider(roleModels), [roleModels]);
  const providers = useMemo(() => ['all', ...groupedModels.map(([provider]) => provider)], [groupedModels]);
  const filteredGroups = useMemo(() => (
    activeProvider === 'all'
      ? groupedModels
      : groupedModels.filter(([provider]) => provider === activeProvider)
  ), [activeProvider, groupedModels]);

  useEffect(() => {
    if (!zdrEnabled || models.length === 0 || activeTab !== 'custom') return;
    const compatibleIds = new Set(models.filter((model) => model.supports_zdr).map((model) => model.id));
    setSelectedCouncil((prev) => prev.filter((id) => compatibleIds.has(id)));
    setSelectedChairman((prev) => {
      if (!prev || compatibleIds.has(prev)) return prev;
      const fallback = models.find((model) => model.supports_zdr && ['chairman', 'both'].includes(model.type));
      return fallback?.id || '';
    });
  }, [activeTab, models, zdrEnabled]);

  const applyPresetToCustom = (preset) => {
    const resolved = resolvePresetModels(preset, models, zdrEnabled || preset.requires_zdr);
    setSelectedPresetId(preset.id);
    setSelectedChairman(resolved.chairman?.id || '');
    setSelectedCouncil(resolved.council.map((model) => model.id));
    setZdrEnabled((current) => current || preset.requires_zdr);
    setActiveTab('custom');
    setCustomRole('council');
    setActiveProvider('all');
  };

  const handlePresetSelect = (preset) => {
    setSelectedPresetId(preset.id);
    if (preset.requires_zdr) setZdrEnabled(true);
  };

  const handleCouncilToggle = (modelId) => {
    setSelectedCouncil((prev) => {
      if (prev.includes(modelId)) return prev.filter((id) => id !== modelId);
      if (prev.length >= MAX_COUNCIL_SIZE) return prev;
      return [...prev, modelId];
    });
  };

  const handleConfirm = () => {
    if (!canConfirm) return;

    onConfirm({
      councilMembers: activeTab === 'presets' ? null : activeCouncil.map((model) => model.id),
      chairmanModel: activeTab === 'presets' ? null : activeChairman.id,
      presetId: activeTab === 'presets' ? selectedPresetId : null,
      zdrEnabled: activeTab === 'presets' ? effectivePresetZdr : zdrEnabled,
      budgetUsd: defaultBudgetUsd,
    });
    onClose();
  };

  const canConfirm = canConfirmModelSelection({
    chairman: activeChairman,
    council: activeCouncil,
    selectedPresetAvailable,
    loading,
    error,
    minCouncilSize: MIN_COUNCIL_SIZE,
  });

  const renderPresetCard = (preset) => {
    const presetZdr = getEffectivePresetZdr(preset, zdrEnabled);
    const resolved = resolvePresetModels(preset, models, presetZdr);
    const availableForZdr = canStartPresetWithZdr(preset, models, presetZdr);
    const isSelected = selectedPresetId === preset.id;
    const cost = estimateSelectionCost({ chairman: resolved.chairman, council: resolved.council });
    const disabledReason = !availableForZdr ? 'Contains models that do not support ZDR' : null;

    return (
      <Card
        key={preset.id}
        role="button"
        tabIndex={availableForZdr ? 0 : -1}
        aria-pressed={isSelected}
        className={cn(
          'p-4 transition-all',
          availableForZdr ? 'cursor-pointer hover:border-primary/50' : 'cursor-not-allowed opacity-50',
          isSelected && 'border-primary bg-primary/5 ring-1 ring-primary',
        )}
        onClick={() => availableForZdr && handlePresetSelect(preset)}
        onKeyDown={(event) => {
          if (availableForZdr && (event.key === 'Enter' || event.key === ' ')) {
            event.preventDefault();
            handlePresetSelect(preset);
          }
        }}
      >
        <div className="flex items-start gap-3">
          <div className={cn(
            'mt-1 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border',
            isSelected && 'border-primary bg-primary text-primary-foreground',
          )}>
            {isSelected && <Check className="h-3 w-3" />}
          </div>
          <div className="min-w-0 flex-1 space-y-3">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div>
                <div className="font-semibold">{preset.label}</div>
                <p className="mt-1 text-sm text-muted-foreground">{preset.description}</p>
              </div>
              <div className="flex flex-wrap gap-1">
                {preset.requires_zdr && <StatPill tone="green"><Lock className="mr-1 h-3 w-3" />ZDR</StatPill>}
                {disabledReason && <StatPill tone="amber">{disabledReason}</StatPill>}
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <StatPill>{resolved.council.length} council</StatPill>
              <StatPill>{formatCost(cost)} est.</StatPill>
              {resolved.hiddenByZdr > 0 && <StatPill tone="amber">{resolved.hiddenByZdr} hidden by ZDR</StatPill>}
            </div>

            <div className="space-y-2">
              <div className="text-xs font-medium text-muted-foreground">Chairman</div>
              <ModelChip model={resolved.chairman} />
              <div className="text-xs font-medium text-muted-foreground">Council</div>
              <div className="flex flex-wrap gap-1.5">
                {resolved.council.map((model) => <ModelChip key={model.id} model={model} />)}
              </div>
            </div>

            {isSelected && (
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={(event) => {
                  event.stopPropagation();
                  applyPresetToCustom(preset);
                }}
              >
                <SlidersHorizontal className="mr-2 h-4 w-4" />
                Edit members
              </Button>
            )}
          </div>
        </div>
      </Card>
    );
  };

  const renderModelCard = (model) => {
    const isChairman = customRole === 'chairman';
    const isSelected = isChairman ? selectedChairman === model.id : selectedCouncil.includes(model.id);
    const isDisabled = !isChairman && !isSelected && selectedCouncil.length >= MAX_COUNCIL_SIZE;

    return (
      <Card
        key={`${customRole}-${model.id}`}
        role="button"
        tabIndex={isDisabled ? -1 : 0}
        aria-pressed={isSelected}
        className={cn(
          'cursor-pointer p-3 transition-all',
          isSelected ? 'border-primary bg-primary/5 ring-1 ring-primary' : 'hover:border-primary/50',
          isDisabled && 'cursor-not-allowed opacity-50',
        )}
        onClick={() => {
          if (isDisabled) return;
          if (isChairman) setSelectedChairman(model.id);
          else handleCouncilToggle(model.id);
        }}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="truncate text-sm font-medium">{getShortName(model.name)}</div>
            <div className="mt-1 text-xs text-muted-foreground">
              ${model.pricing.input}/M in / ${model.pricing.output}/M out
            </div>
          </div>
          <div className={cn(
            'flex h-5 w-5 shrink-0 items-center justify-center rounded border',
            isChairman && 'rounded-full',
            isSelected && 'border-primary bg-primary text-primary-foreground',
          )}>
            {isSelected && <Check className="h-3 w-3" />}
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-1">
          {model.supports_zdr && <StatPill tone="green">ZDR</StatPill>}
          {(model.capabilities || []).slice(0, 3).map((capability) => (
            <StatPill key={capability}>{capability}</StatPill>
          ))}
        </div>
      </Card>
    );
  };

  return (
    <TooltipProvider>
      <Dialog open={isOpen} onOpenChange={onClose}>
        <DialogContent className="flex max-h-[90vh] max-w-5xl flex-col p-0">
          <DialogHeader className="border-b px-6 py-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <DialogTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-primary" />
                New conversation
              </DialogTitle>
              <DialogDescription className="sr-only">
                Choose a preset or customize the chairman, council members, routing privacy, and estimated cost for a new conversation.
              </DialogDescription>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    type="button"
                    variant={effectivePresetZdr ? 'default' : 'outline'}
                    size="sm"
                    aria-disabled={zdrToggleLocked}
                    className={cn(zdrToggleLocked && 'cursor-not-allowed')}
                    onClick={() => {
                      if (zdrToggleLocked) return;
                      setZdrEnabled((value) => !value);
                    }}
                  >
                    <Lock className="mr-2 h-4 w-4" />
                    ZDR only
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {zdrToggleLocked
                    ? 'This preset requires Zero Data Retention model routes.'
                    : 'Restrict this conversation to Zero Data Retention model routes.'}
                </TooltipContent>
              </Tooltip>
            </div>
          </DialogHeader>

          <div className="border-b px-6 py-3">
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                size="sm"
                variant={activeTab === 'presets' ? 'default' : 'outline'}
                onClick={() => setActiveTab('presets')}
              >
                Presets
              </Button>
              <Button
                type="button"
                size="sm"
                variant={activeTab === 'custom' ? 'default' : 'outline'}
                onClick={() => setActiveTab('custom')}
              >
                Custom
              </Button>
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-hidden">
            <ScrollArea className="h-[58vh] px-6 py-4">
              {loading ? (
                <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">Loading models...</div>
              ) : error ? (
                <div className="text-center text-destructive">{error}</div>
              ) : activeTab === 'presets' ? (
                <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                  {presets.map(renderPresetCard)}
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex flex-wrap gap-2">
                      <Button
                        type="button"
                        size="sm"
                        variant={customRole === 'council' ? 'default' : 'outline'}
                        onClick={() => {
                          setCustomRole('council');
                          setActiveProvider('all');
                        }}
                      >
                        <Users className="mr-2 h-4 w-4" />
                        Council members
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant={customRole === 'chairman' ? 'default' : 'outline'}
                        onClick={() => {
                          setCustomRole('chairman');
                          setActiveProvider('all');
                        }}
                      >
                        Chairman
                      </Button>
                    </div>
                    <StatPill tone={selectedCouncil.length >= MIN_COUNCIL_SIZE ? 'green' : 'amber'}>
                      {selectedCouncil.length}/{MAX_COUNCIL_SIZE} council selected
                    </StatPill>
                  </div>

                  <div className="flex gap-2 overflow-x-auto pb-1">
                    {providers.map((provider) => {
                      const count = provider === 'all'
                        ? roleModels.length
                        : groupedModels.find(([name]) => name === provider)?.[1]?.length || 0;
                      return (
                        <Button
                          key={provider}
                          type="button"
                          variant={activeProvider === provider ? 'default' : 'outline'}
                          size="sm"
                          className="shrink-0"
                          onClick={() => setActiveProvider(provider)}
                        >
                          {provider !== 'all' && (
                            <span className={cn('mr-2 h-2 w-2 rounded-full', PROVIDER_COLORS[provider] || PROVIDER_COLORS.Other)} />
                          )}
                          {provider === 'all' ? 'All' : provider}
                          <span className="ml-1.5 text-xs opacity-70">{count}</span>
                        </Button>
                      );
                    })}
                  </div>

                  {filteredGroups.map(([provider, providerModels]) => (
                    <div key={provider} className="space-y-3">
                      <div className="flex items-center gap-2">
                        <span className={cn('h-3 w-3 rounded-full', PROVIDER_COLORS[provider] || PROVIDER_COLORS.Other)} />
                        <span className="font-semibold">{provider}</span>
                        <span className="text-xs text-muted-foreground">({providerModels.length})</span>
                      </div>
                      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
                        {providerModels.map(renderModelCard)}
                      </div>
                      <Separator />
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </div>

          <DialogFooter className="border-t bg-muted/10 px-6 py-4">
            <div className="flex w-full flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <div className="flex flex-wrap gap-2">
                <StatPill>{activeCouncil.length} council</StatPill>
                <StatPill>{formatCost(estimatedCost)} est.</StatPill>
                <StatPill tone={effectivePresetZdr ? 'green' : 'default'}>
                  {effectivePresetZdr ? 'ZDR on' : 'Standard routing'}
                </StatPill>
                {defaultBudgetUsd != null && <StatPill>Budget ${defaultBudgetUsd}</StatPill>}
              </div>
              <div className="flex gap-2">
                <Button type="button" variant="outline" onClick={onClose}>Cancel</Button>
                <Button type="button" onClick={handleConfirm} disabled={!canConfirm}>
                  Start conversation
                </Button>
              </div>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </TooltipProvider>
  );
}
