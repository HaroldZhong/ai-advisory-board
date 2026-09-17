import { useEffect, useRef, useState } from 'react';
import { Sheet, SheetContent, SheetTitle, SheetDescription } from './ui/sheet';
import { Button } from './ui/button';
import { FileText, Plus, PanelRightClose, CheckCircle2, CircleHelp, AlertTriangle } from 'lucide-react';
import { getTurnPresentation } from '@/utils/evidence';
import { formatCurrency } from '@/utils/trustState';

export function TurnState({ message, onOpen }) {
  const meta = message.metadata || {};
  const { state, invalid, incomplete, needsAttention, settled, label } = getTurnPresentation(message);
  const statusText = `Answer: ${state.generation} · Saved: ${state.persistence} · Memory: ${state.memory}`;
  return (
    <div className="flex items-start justify-between gap-3 text-xs text-muted-foreground">
      <details open={needsAttention} className="min-w-0 flex-1">
        <summary title={statusText} className="flex w-fit cursor-pointer list-none items-center gap-1.5 rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring [&::-webkit-details-marker]:hidden">
          {needsAttention ? <AlertTriangle className="h-3.5 w-3.5 text-amber-600" /> : settled ? <CheckCircle2 className="h-3.5 w-3.5" /> : <CircleHelp className="h-3.5 w-3.5" />}
          <span role="status">{label}</span>
        </summary>
        <div className="mt-2 space-y-1 border-l-2 pl-3 leading-relaxed">
          <p>{statusText}</p>
          <p>Recorded cost: {message.running_cost == null ? 'Unknown' : formatCurrency(message.running_cost)}. Unreported charges are not included.</p>
          {message.interruption && <p>{message.interruption}. Upstream cancellation and final charges are unconfirmed.</p>}
          {message.unconfirmed && <p>This text is unconfirmed. Copy it before leaving.</p>}
          {invalid.length > 0 && <p>Unverified citations: {invalid.map((item) => item.token).join(', ')}</p>}
          {incomplete && <p>Some source content was not read. Open Sources for coverage.</p>}
          {state.memory === 'failed' && <p>Memory indexing failed. Check the separate saved status above.</p>}
        </div>
      </details>
      {meta.evidence_snapshot?.sources?.length > 0 && (
        <button type="button" className="shrink-0 underline underline-offset-2" onClick={(event) => onOpen(null, meta.evidence_snapshot, event.currentTarget)}>
          Sources · {meta.evidence_snapshot.sources.length}
        </button>
      )}
    </div>
  );
}

export default function EvidencePanel({
  open, onClose, selection, view, onViewChange, sources, selectedIds, onSelectionChange,
  disabled, onAddFiles, uploadDisabled, isUploading, triggerRef, returnFocusRef,
}) {
  const [narrow, setNarrow] = useState(() => window.matchMedia('(max-width: 1023px)').matches);
  const panel = useRef(null);
  useEffect(() => {
    const query = window.matchMedia('(max-width: 1023px)');
    const update = () => setNarrow(query.matches);
    query.addEventListener('change', update);
    return () => query.removeEventListener('change', update);
  }, []);
  useEffect(() => {
    if (!narrow && open && selection) panel.current?.focus();
  }, [selection, narrow, open]);
  const restoreFocus = () => {
    const trigger = returnFocusRef.current;
    if (trigger?.isConnected) trigger.focus(); else triggerRef.current?.focus();
  };
  const close = () => { onClose(); restoreFocus(); };
  const { token, snapshot } = selection || {};
  const selectedCount = sources.filter((source) => selectedIds === null || selectedIds.includes(source.attachment_id)).length;
  const body = (
    <div className="flex min-h-0 flex-1 flex-col">
      <div role="group" aria-label="Materials view" className="mx-4 mb-3 grid shrink-0 grid-cols-2 gap-1 rounded-md bg-muted p-1">
        <Button variant={view === 'materials' ? 'secondary' : 'ghost'} className={view === 'materials' ? 'bg-background shadow-sm' : ''} size="sm" aria-pressed={view === 'materials'} onClick={() => onViewChange('materials')}>Materials</Button>
        <Button variant={view === 'sources' ? 'secondary' : 'ghost'} className={view === 'sources' ? 'bg-background shadow-sm' : ''} size="sm" aria-pressed={view === 'sources'} onClick={() => onViewChange('sources')} disabled={!snapshot}>Answer sources</Button>
      </div>
      <div hidden={view !== 'materials'} className="mt-0 min-h-0 flex-1 overflow-y-auto px-4 pb-4">
        <div className="mb-4 space-y-2">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-sm font-medium">Conversation files</h3>
            <Button variant="ghost" size="sm" className="h-8 px-2" disabled={uploadDisabled} onClick={onAddFiles}>
              <Plus className="mr-1 h-3.5 w-3.5" />Add files
            </Button>
          </div>
          <p className="text-xs leading-relaxed text-muted-foreground">Select files for your next answer or Council.</p>
        </div>
        {isUploading && <p role="status" className="mb-3 text-xs text-muted-foreground">Uploading and extracting files…</p>}
        {sources.length ? <fieldset disabled={disabled}>
          <legend className="sr-only">Choose materials for the next answer</legend>
          <div className="mb-2 flex items-center justify-between border-b pb-3 text-xs">
            <span>{selectedCount} of {sources.length} selected</span>
            <div className="flex gap-3">
              <button type="button" className="underline underline-offset-4 disabled:opacity-50" onClick={() => onSelectionChange(null)}>Use all</button>
              <button type="button" className="underline underline-offset-4 disabled:opacity-50" onClick={() => onSelectionChange([])}>Use none</button>
            </div>
          </div>
          <div className="space-y-1">
            {sources.map((source) => <label key={source.attachment_id} className="flex cursor-pointer items-start gap-3 rounded-md px-2 py-3 hover:bg-muted/70 has-[:disabled]:cursor-default has-[:disabled]:opacity-60">
              <input type="checkbox" className="mt-1 h-4 w-4 shrink-0 accent-current" aria-label={`Use ${source.filename}`} checked={selectedIds === null || selectedIds.includes(source.attachment_id)} onChange={(event) => {
                const selected = new Set(selectedIds ?? sources.map((item) => item.attachment_id));
                if (event.target.checked) selected.add(source.attachment_id); else selected.delete(source.attachment_id);
                onSelectionChange([...selected]);
              }} />
              <FileText className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
              <span className="min-w-0 space-y-1">
                <span className="block break-words text-sm leading-snug">{source.filename}</span>
                <span className="block text-xs text-muted-foreground">{{ success: 'Text extracted', partial: 'Partially extracted', failed: 'Extraction failed', processing: 'Extracting…' }[source.status] || 'Rechecked on send'}</span>
                {source.warning && <span className="block break-words text-xs text-amber-700 dark:text-amber-400">{source.warning}</span>}
              </span>
            </label>)}
          </div>
          <p role="status" className="mt-4 border-t pt-3 text-xs leading-relaxed text-muted-foreground">
            {selectedIds === null ? 'All files, including new uploads, will be used.' : selectedCount ? 'Only selected files will be used. Select new uploads to include them.' : 'The next answer will not use conversation files.'}
          </p>
        </fieldset> : <div className="border-t py-8 text-center">
          <FileText className="mx-auto mb-3 h-6 w-6 text-muted-foreground" />
          <p className="text-sm font-medium">Bring your sources</p>
          <p className="mt-2 text-xs leading-relaxed text-muted-foreground">Add a document or drop files into the conversation. Your source list will appear here.</p>
        </div>}
      </div>
      <div hidden={view !== 'sources'} className="mt-0 min-h-0 flex-1 space-y-4 overflow-y-auto break-words px-4 pb-4 text-sm">
        <p className="text-xs leading-relaxed text-muted-foreground">Sources saved with this answer. Check each excerpt against the claim.</p>
        {snapshot?.sources.map((source) => (
          <section key={source.source_id} className="space-y-2 border-t pt-3">
            <h3 className="font-medium">{source.alias}: {source.title}</h3>
            <p className="text-xs text-muted-foreground">{source.included?.length || 0} excerpts used{source.omitted?.length ? ` · ${source.omitted.length} not read` : ''}{!['success', 'partial'].includes(source.status) ? ` · ${source.status}` : ''}</p>
            {source.warning && <p>{source.warning}</p>}
            {source.status === 'partial' && <p className="text-xs text-amber-700 dark:text-amber-400">Extraction is incomplete. Only extracted text is included.</p>}
            <details className="text-xs text-muted-foreground">
              <summary className="cursor-pointer">Source details</summary>
              <p className="break-all">Version: {source.version_id || 'unavailable'}</p>
              <p>Saved with this answer; later file selections do not change it. A matching citation is not proof that the claim is supported.</p>
            </details>
            {Object.entries(snapshot.citations).filter(([, chunk]) => chunk.source_id === source.source_id).map(([id, chunk]) => (
              <blockquote key={id} tabIndex={-1} ref={(element) => { if (id === token && element) element.scrollIntoView({ block: 'nearest' }); }}
                className={`whitespace-pre-wrap border-l-2 px-3 py-2 leading-relaxed ${id === token ? 'border-primary bg-primary/10 ring-1 ring-primary/30' : 'border-muted'}`}>
                <p className="mb-2 text-xs font-medium" title={`Characters ${chunk.start}–${chunk.end}`}>{id} · {chunk.page ? `Page ${chunk.page} · ` : ''}Excerpt {chunk.ordinal}</p>
                {chunk.text}
              </blockquote>
            ))}
          </section>
        ))}
      </div>
    </div>
  );
  if (!open) return null;
  return narrow ? (
    <Sheet open onOpenChange={(isOpen) => { if (!isOpen) onClose(); }}>
      <SheetContent side="right" className="flex h-dvh w-full flex-col gap-0 p-0 sm:max-w-[380px]" onCloseAutoFocus={(event) => { event.preventDefault(); restoreFocus(); }}>
        <SheetTitle className="flex h-14 shrink-0 items-center px-4 text-sm">Materials</SheetTitle>
        <SheetDescription className="sr-only">Conversation files and sources used in an answer</SheetDescription>
        {body}
      </SheetContent>
    </Sheet>
  ) : (
    <aside id="materials-sidebar" ref={panel} tabIndex={-1} aria-label="Materials" onKeyDown={(event) => { if (event.key === 'Escape') { event.stopPropagation(); close(); } }}
      className="flex h-full w-[340px] shrink-0 flex-col border-l bg-muted/10 outline-none">
      <div className="flex h-14 shrink-0 items-center justify-between px-4">
        <h2 className="text-sm font-semibold">Materials</h2>
        <Button variant="ghost" size="icon" className="h-8 w-8" aria-label="Close materials" onClick={close}><PanelRightClose className="h-4 w-4" /></Button>
      </div>
      {body}
    </aside>
  );
}


export function UnconfirmedDrafts({ drafts, onDiscard }) {
  useEffect(() => {
    if (!drafts.length) return undefined;
    const warn = (event) => { event.preventDefault(); event.returnValue = ''; };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [drafts.length]);
  if (!drafts.length) return null;
  return (
    <details className="max-h-[35vh] w-full shrink-0 overflow-y-auto border-t bg-background px-4 py-2">
      <summary className="cursor-pointer font-medium">Unconfirmed drafts ({drafts.length})</summary>
      <p className="my-2 text-xs text-muted-foreground">Kept in this tab until you discard them. Leaving or reloading the tab will ask for confirmation.</p>
      {drafts.map((draft) => {
        const message = draft.message;
        const text = [
          `Question: ${draft.prompt}`,
          message.content,
          ...(message.stage1 || []).map((result) => `${result.model}: ${result.response}`),
          ...(message.stage2 || []).map((result) => `${result.model}: ${result.ranking}`),
          message.stage3?.response,
        ].filter(Boolean).join('\n\n');
        return <section key={draft.id} className="space-y-2 border-t py-2">
          <p className="text-xs">Conversation {draft.conversationId.slice(0, 8)} · Save unconfirmed</p>
          <textarea aria-label="Unconfirmed response text" readOnly value={text} onFocus={(event) => event.target.select()} rows={8} className="w-full resize-y rounded border bg-background p-2 text-sm" />
          <button type="button" className="text-sm underline" onClick={() => onDiscard(draft.id)}>Discard this draft</button>
        </section>;
      })}
    </details>
  );
}
