import { useMemo, useState } from 'react';
import { ArrowLeft, ArrowRight, CheckCircle2, DollarSign, KeyRound, Loader2, ShieldCheck } from 'lucide-react';

import { api } from '@/api';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import {
  FIRST_RUN_BUDGET_PRESETS,
  buildFirstRunSettings,
  looksLikeOpenRouterKey,
} from '@/utils/firstRunSetup';

const STEPS = [
  { id: 'connect', label: 'Connect' },
  { id: 'privacy', label: 'Privacy' },
  { id: 'budget', label: 'Budget' },
];

function ChoiceButton({ selected, children, className = '', ...props }) {
  return (
    <button
      type="button"
      className={[
        'w-full rounded-lg border p-4 text-left transition-colors',
        selected
          ? 'border-primary bg-primary/10 text-foreground'
          : 'border-border bg-background hover:bg-muted/60',
        className,
      ].join(' ')}
      aria-pressed={selected}
      {...props}
    >
      {children}
    </button>
  );
}

export default function FirstRunSetup({ isOpen, onComplete }) {
  const [step, setStep] = useState(0);
  const [apiKey, setApiKey] = useState('');
  const [zdrChoice, setZdrChoice] = useState(null);
  const [budgetUsd, setBudgetUsd] = useState(2);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');

  const trimmedApiKey = apiKey.trim();
  const keyIsValid = looksLikeOpenRouterKey(trimmedApiKey);
  const currentStep = STEPS[step];

  const canContinue = useMemo(() => {
    if (currentStep.id === 'connect') return keyIsValid;
    if (currentStep.id === 'privacy') return Boolean(zdrChoice);
    return true;
  }, [currentStep.id, keyIsValid, zdrChoice]);

  const handleNext = async () => {
    setError('');

    if (step < STEPS.length - 1) {
      setStep((value) => value + 1);
      return;
    }

    setIsSaving(true);
    try {
      await api.setupConfig(trimmedApiKey);
      onComplete(buildFirstRunSettings({ zdrChoice, budgetUsd }));
    } catch (err) {
      setError(err?.message || 'Failed to save setup.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={() => {}}>
      <DialogContent className="sm:max-w-[560px] [&>button]:hidden" aria-describedby="first-run-description">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <KeyRound className="h-5 w-5 text-primary" />
            Set Up AI Advisory Board
          </DialogTitle>
          <DialogDescription id="first-run-description">
            Connect OpenRouter, choose your default privacy mode, and set a starting session budget.
          </DialogDescription>
        </DialogHeader>

        <div className="flex items-center gap-2" aria-label="Setup progress">
          {STEPS.map((item, index) => (
            <div key={item.id} className="flex min-w-0 flex-1 items-center gap-2">
              <div
                className={[
                  'flex h-7 w-7 shrink-0 items-center justify-center rounded-full border text-xs font-semibold',
                  index <= step ? 'border-primary bg-primary text-primary-foreground' : 'border-border bg-muted',
                ].join(' ')}
                aria-current={index === step ? 'step' : undefined}
              >
                {index < step ? <CheckCircle2 className="h-4 w-4" /> : index + 1}
              </div>
              <span className="truncate text-xs font-medium text-muted-foreground">{item.label}</span>
              {index < STEPS.length - 1 && <div className="hidden h-px flex-1 bg-border sm:block" />}
            </div>
          ))}
        </div>

        <div className="min-h-[280px] py-2">
          {currentStep.id === 'connect' && (
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium" htmlFor="openrouter-api-key">
                  OpenRouter API key
                </label>
                <Input
                  id="openrouter-api-key"
                  type="password"
                  value={apiKey}
                  onChange={(event) => setApiKey(event.target.value)}
                  placeholder="sk-or-v1-..."
                  autoFocus
                  autoComplete="off"
                />
                <p className="text-xs text-muted-foreground">
                  The key is sent to the local backend and is not stored in browser settings.
                </p>
              </div>
              {trimmedApiKey && !keyIsValid && (
                <p className="text-sm text-destructive" role="alert">
                  Enter an OpenRouter key that starts with sk-or-.
                </p>
              )}
            </div>
          )}

          {currentStep.id === 'privacy' && (
            <div className="space-y-3">
              <ChoiceButton selected={zdrChoice === 'on'} onClick={() => setZdrChoice('on')}>
                <div className="flex items-start gap-3">
                  <ShieldCheck className="mt-0.5 h-5 w-5 text-green-600" />
                  <div>
                    <div className="font-medium">Private routing by default</div>
                    <div className="mt-1 text-sm text-muted-foreground">
                      Use OpenRouter Zero Data Retention endpoints when available.
                    </div>
                  </div>
                </div>
              </ChoiceButton>

              <ChoiceButton selected={zdrChoice === 'off'} onClick={() => setZdrChoice('off')}>
                <div className="flex items-start gap-3">
                  <ShieldCheck className="mt-0.5 h-5 w-5 text-muted-foreground" />
                  <div>
                    <div className="font-medium">Standard routing by default</div>
                    <div className="mt-1 text-sm text-muted-foreground">
                      Keep the widest model selection and turn privacy mode on when needed.
                    </div>
                  </div>
                </div>
              </ChoiceButton>
            </div>
          )}

          {currentStep.id === 'budget' && (
            <div className="space-y-3">
              <div className="grid gap-3 sm:grid-cols-2">
                {FIRST_RUN_BUDGET_PRESETS.map((preset) => (
                  <ChoiceButton
                    key={preset.id}
                    selected={budgetUsd === preset.value}
                    onClick={() => setBudgetUsd(preset.value)}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="flex items-center gap-2 font-medium">
                          <DollarSign className="h-4 w-4 text-primary" />
                          {preset.label}
                        </div>
                        <div className="mt-1 text-sm text-muted-foreground">{preset.description}</div>
                      </div>
                      {preset.recommended && (
                        <span className="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
                          Recommended
                        </span>
                      )}
                    </div>
                  </ChoiceButton>
                ))}
              </div>
            </div>
          )}
        </div>

        {error && (
          <p className="text-sm text-destructive" role="alert" aria-live="polite">
            {error}
          </p>
        )}

        <DialogFooter className="gap-2 sm:justify-between">
          <Button
            type="button"
            variant="outline"
            onClick={() => setStep((value) => Math.max(0, value - 1))}
            disabled={step === 0 || isSaving}
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
          <Button type="button" onClick={handleNext} disabled={!canContinue || isSaving}>
            {isSaving ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Saving
              </>
            ) : step === STEPS.length - 1 ? (
              <>
                Finish
                <CheckCircle2 className="ml-2 h-4 w-4" />
              </>
            ) : (
              <>
                Continue
                <ArrowRight className="ml-2 h-4 w-4" />
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
