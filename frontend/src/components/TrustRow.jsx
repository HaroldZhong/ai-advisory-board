import { AlertTriangle, DollarSign, FileText, Globe, Settings, Shield, ShieldOff, Users } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { getTrustRowCostTileClass, getTrustRowGridClass } from '@/utils/responsiveChatLayout';
import { formatTrustRowState, getEffectiveBudgetWarning } from '@/utils/trustState';

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
      title={title}
      aria-label={title || `${label}: ${detail}`}
      className={cn(
        'min-h-[52px] rounded-md border px-3 py-2 text-left transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
        toneClasses[tone] || toneClasses.neutral,
        onClick && !disabled && 'hover:border-primary/50 hover:bg-muted/60',
        disabled && 'cursor-not-allowed opacity-70',
      )}
    >
      <div className="flex items-center gap-2">
        <Icon className="h-4 w-4 shrink-0" />
        <div className="min-w-0">
          <div className="truncate text-xs font-semibold">{label}</div>
          <div className="truncate text-[11px] text-muted-foreground">{detail}</div>
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
    <div className="mt-2 h-1.5 rounded-full bg-background/80">
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
  privacyDisabled = false,
  privacyDisabledReason,
}) {
  const state = formatTrustRowState({ conversation, settings, attachmentCount });
  const warning = getEffectiveBudgetWarning(state.budget.spentPct, budgetWarning?.threshold);
  const warningTone = warning?.level === 'danger' ? 'danger' : warning?.level === 'warn' ? 'warn' : 'caution';
  const privacyIcon = state.privacy.effectiveZdr ? Shield : ShieldOff;

  const handlePrivacyToggle = () => {
    if (state.privacy.locked || privacyDisabled || !onUpdateConversationPrivacy) return;
    onUpdateConversationPrivacy(!state.privacy.effectiveZdr);
  };

  return (
    <div className="space-y-2">
      <div className={getTrustRowGridClass()}>
        <TrustTile
          icon={Users}
          label={state.council.label}
          detail={state.council.detail}
          title={`Council: ${state.council.label}, ${state.council.detail}`}
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
          label={state.budget.label}
          detail={state.budget.detail}
          onClick={onOpenBudget}
          tone={state.budget.tone}
          title="Open session budget settings"
        >
          <BudgetProgress spentPct={state.budget.spentPct} tone={state.budget.tone} />
        </TrustTile>

        <div
          className={cn(
            'min-h-[52px] rounded-md border px-3 py-2 transition-colors',
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
              <div className="truncate text-[11px] text-muted-foreground">{state.tools.detail}</div>
            </div>
          </button>
          {state.tools.webEnabled && (
            <button
              type="button"
              className="mt-1 rounded px-1.5 py-0.5 text-[10px] font-medium uppercase text-muted-foreground hover:bg-background/80 hover:text-foreground"
              onClick={onToggleWebDepth}
              title={`Currently ${state.tools.webDepth}. Click to toggle depth.`}
            >
              {state.tools.webDepth}
            </button>
          )}
        </div>

        <div className={getTrustRowCostTileClass()}>
          <div className="text-[11px] text-muted-foreground">{state.cost.label}</div>
          <div className="font-mono text-sm font-semibold">{state.cost.value}</div>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 lg:hidden"
            onClick={onOpenAdvancedSettings}
            title="Advanced settings"
          >
            <Settings className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      <div className="hidden justify-end lg:flex">
        <Button
          variant="ghost"
          size="sm"
          className="h-7 px-2 text-xs"
          onClick={onOpenAdvancedSettings}
        >
          <Settings className="mr-1 h-3.5 w-3.5" />
          Advanced
        </Button>
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
