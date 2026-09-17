import { useState } from 'react';
import { AlertTriangle, BrainCircuit, Check, DollarSign, FileText, Globe, Settings, Shield, ShieldOff, Users } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Popover, PopoverTrigger, PopoverContent } from '@/components/ui/popover';
import { cn } from '@/lib/utils';
import { getTrustRowCostTileClass, getTrustRowGridClass } from '@/utils/responsiveChatLayout';
import { formatTrustRowState, getEffectiveBudgetWarning } from '@/utils/trustState';
import {
  THINKING_EFFORT_LEVELS,
  getThinkingEffortOption,
} from '@/utils/thinkingEffort';

const toneClasses = {
  neutral: 'border-border bg-muted/30 text-foreground',
  caution: 'border-yellow-500/30 bg-yellow-500/10 text-yellow-800 dark:text-yellow-300',
  warn: 'border-orange-500/35 bg-orange-500/10 text-orange-800 dark:text-orange-300',
  danger: 'border-red-500/40 bg-red-500/10 text-red-800 dark:text-red-300',
};

function TrustTile({
  icon: Icon,
  label,
  detail,
  onClick,
  disabled = false,
  tone = 'neutral',
  title,
  children,
}) {
  const Component = onClick ? 'button' : 'div';

  return (
    <Component
      type={onClick ? 'button' : undefined}
      onClick={disabled ? undefined : onClick}
      disabled={onClick ? disabled : undefined}
      title={title ? `${title}. ${detail}` : detail}
      aria-label={title || `${label}: ${detail}`}
      className={cn(
        'relative flex h-8 min-w-0 max-w-[190px] items-center overflow-hidden rounded-md border px-2 text-left transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
        toneClasses[tone] || toneClasses.neutral,
        onClick && !disabled && 'hover:border-primary/50 hover:bg-muted/60',
        disabled && 'cursor-not-allowed opacity-70',
      )}
    >
      <div className="flex min-w-0 items-center gap-2">
        <Icon className="h-4 w-4 shrink-0" />
        <div className="min-w-0">
          <div className="truncate text-xs font-semibold">{label}</div>
          <div className={children ? 'text-[10px] leading-3 text-muted-foreground' : 'sr-only'}>{detail}</div>
        </div>
      </div>
      {children}
    </Component>
  );
}

function BudgetProgress({ spentPct, tone }) {
  if (spentPct == null) return null;
  const width = `${Math.min(Math.max(spentPct, 0), 1) * 100}%`;

  return (
    <div role="progressbar" aria-label="Session budget used" aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={Math.round(Math.min(Math.max(spentPct, 0), 1) * 100)} aria-valuetext={`${Math.round(spentPct * 100)}% used`}
      className="absolute inset-x-0 bottom-0 h-0.5 bg-background/80">
      <div
        className={cn(
          'h-full rounded-full',
          tone === 'danger' && 'bg-red-500',
          tone === 'warn' && 'bg-orange-500',
          tone === 'caution' && 'bg-yellow-500',
          tone === 'neutral' && 'bg-green-500',
        )}
        style={{ width }}
      />
    </div>
  );
}

export default function TrustRow({
  conversation,
  settings,
  attachmentCount = 0,
  budgetWarning,
  onOpenBudget,
  onToggleWebSearch,
  onToggleWebDepth,
  onOpenAdvancedSettings,
  onUpdateConversationPrivacy,
  onUpdateThinkingEffort,
  privacyDisabled = false,
  privacyDisabledReason,
  thinkingDisabled = false,
  thinkingDisabledReason,
  zdrAvailable = true,
}) {
  const [isThinkingMenuOpen, setIsThinkingMenuOpen] = useState(false);
  const state = formatTrustRowState({ conversation, settings, attachmentCount, zdrAvailable });
  const warning = getEffectiveBudgetWarning(
    state.budget.spentPct,
    budgetWarning?.threshold,
    state.budget.notifyThresholds,
  );
  const warningTone = warning?.level === 'danger' ? 'danger' : warning?.level === 'warn' ? 'warn' : 'caution';
  const privacyIcon = state.privacy.effectiveZdr ? Shield : ShieldOff;
  const thinkingCanUpdate = Boolean(conversation?.id && onUpdateThinkingEffort) && !thinkingDisabled;
  // v1.3.0 B3: every conversation can select every level -- the preset no longer caps.
  const selectableThinkingEfforts = THINKING_EFFORT_LEVELS;

  const handlePrivacyToggle = () => {
    if (state.privacy.locked || privacyDisabled || !onUpdateConversationPrivacy) return;
    onUpdateConversationPrivacy(!state.privacy.effectiveZdr);
  };

  const handleThinkingSelect = (effort) => {
    if (!thinkingCanUpdate || effort === state.thinking.value) {
      setIsThinkingMenuOpen(false);
      return;
    }
    onUpdateThinkingEffort(effort);
    setIsThinkingMenuOpen(false);
  };

  return (
    <div className="space-y-2">
      <div className={getTrustRowGridClass()}>
        <TrustTile
          icon={Users}
          label={state.council.label === 'Chat' ? state.council.detail.split('/').pop() : state.council.label}
          detail={state.council.detail}
          title={`${state.council.label}: ${state.council.detail}`}
        />

        <TrustTile
          icon={privacyIcon}
          label={state.privacy.label}
          detail={state.privacy.detail}
          onClick={handlePrivacyToggle}
          disabled={state.privacy.locked || privacyDisabled}
          tone={state.privacy.effectiveZdr ? 'caution' : 'neutral'}
          title={
            state.privacy.locked
              ? 'Private preset requires Zero Data Retention'
              : privacyDisabled
                ? privacyDisabledReason || 'Privacy changes are temporarily disabled'
              : 'Toggle Zero Data Retention for this conversation'
          }
        />

        <TrustTile
          icon={DollarSign}
          label={state.budget.budgetUsd == null ? `${state.cost.value} · No limit` : state.budget.label}
          detail={state.budget.detail}
          onClick={onOpenBudget}
          tone={state.budget.tone}
          title="Open session budget settings"
        >
          <BudgetProgress spentPct={state.budget.spentPct} tone={state.budget.tone} />
        </TrustTile>

        <Popover open={isThinkingMenuOpen && thinkingCanUpdate} onOpenChange={setIsThinkingMenuOpen}>
          <PopoverTrigger asChild>
            <button type="button" disabled={!thinkingCanUpdate}
              aria-label={`Thinking: ${state.thinking.label}`}
              title={thinkingDisabled ? thinkingDisabledReason : state.thinking.detail}
              className={cn('flex h-8 items-center gap-2 rounded-md border px-2 text-xs font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-70', toneClasses[state.thinking.tone] || toneClasses.neutral)}>
              <BrainCircuit className="h-4 w-4" />{state.thinking.label}
            </button>
          </PopoverTrigger>
          <PopoverContent side="top" align="start" className="w-[340px] p-2">
              <div className="px-2 pb-2">
                <div className="text-xs font-semibold">Thinking effort</div>
                <div className="text-[11px] text-muted-foreground">
                  Applies to future turns on supported models.
                </div>
              </div>
              <div className="space-y-1">
                {selectableThinkingEfforts.map((effort) => {
                  const option = getThinkingEffortOption(effort);
                  const selected = effort === state.thinking.value;

                  return (
                    <button
                      key={effort}
                      type="button"
                      className={cn(
                        'flex w-full items-start gap-2 rounded px-2 py-2 text-left text-sm hover:bg-muted',
                        selected && 'bg-muted',
                      )}
                      onClick={() => handleThinkingSelect(effort)}
                    >
                      <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center">
                        {selected && <Check className="h-3.5 w-3.5" />}
                      </span>
                      <span className="min-w-0">
                        <span className="block font-medium">{option.label}</span>
                        <span className="block text-xs text-muted-foreground">{option.description}</span>
                        <span className="block text-[11px] text-muted-foreground">{option.costHint}</span>
                      </span>
                    </button>
                  );
                })}
              </div>
          </PopoverContent>
        </Popover>

        <div
          className={cn(
            'flex h-8 items-center gap-1 rounded-md border px-2 transition-colors',
            toneClasses[state.tools.webEnabled ? 'caution' : 'neutral'],
          )}
        >
          <button
            type="button"
            className="flex w-full items-center gap-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            onClick={onToggleWebSearch}
            title={state.tools.webEnabled ? 'Disable web search' : 'Enable web search'}
            aria-label={`${state.tools.label}: ${state.tools.detail}`}
          >
            {state.tools.attachmentCount > 0 ? (
              <FileText className="h-4 w-4 shrink-0" />
            ) : (
              <Globe className="h-4 w-4 shrink-0" />
            )}
            <div className="min-w-0">
              <div className="truncate text-xs font-semibold">{state.tools.label}</div>
              <div className="sr-only">{state.tools.detail}</div>
            </div>
          </button>
          {state.tools.webEnabled && (
            <button
              type="button"
              className="rounded px-1.5 py-0.5 text-[10px] font-medium uppercase text-muted-foreground hover:bg-background/80 hover:text-foreground"
              onClick={onToggleWebDepth}
              title={`Currently ${state.tools.webDepth}. Click to toggle depth.`}
            >
              {state.tools.webDepth}
            </button>
          )}
        </div>

        <div className={getTrustRowCostTileClass()}>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={onOpenAdvancedSettings}
            title="Advanced settings"
            aria-label="Advanced settings"
          >
            <Settings className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {warning && (
        <div
          role={warning.level === 'danger' ? 'alert' : 'status'}
          aria-live={warning.level === 'danger' ? 'assertive' : 'polite'}
          className={cn(
            'flex items-center justify-between gap-3 rounded-md border px-3 py-2 text-sm',
            toneClasses[warningTone],
          )}
        >
          <div className="flex min-w-0 items-center gap-2">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            <div className="min-w-0">
              <div className="font-semibold">{warning.label}</div>
              <div className="text-xs text-muted-foreground">{warning.body}</div>
            </div>
          </div>
          <Button variant="outline" size="sm" className="shrink-0" onClick={onOpenBudget}>
            {warning.action}
          </Button>
        </div>
      )}
    </div>
  );
}
